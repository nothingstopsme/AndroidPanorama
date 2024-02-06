package nothingstopsme.panorama

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.opencv.global.opencv_calib3d
import org.bytedeco.opencv.global.opencv_core
import org.bytedeco.opencv.opencv_calib3d.CvLevMarq
import org.bytedeco.opencv.opencv_core.CvMat
import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.opencv_core.MatExpr
import org.bytedeco.opencv.opencv_core.Scalar
import org.bytedeco.opencv.opencv_core.TermCriteria
import org.bytedeco.opencv.opencv_stitching.CameraParamsVector
import org.bytedeco.opencv.opencv_stitching.ImageFeaturesVector
import org.bytedeco.opencv.opencv_stitching.MatchesInfoVector
import kotlin.coroutines.coroutineContext
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max


class SequentialRotationBundleAdjuster(private val context: Context) {

    private data class SimplifiedPinholeModel(val invK: Mat, val R: Mat)

    private data class Correspondence(val index:Long, val keypoints: Mat)

    private data class ModelDescriptor(val referenceIndex:Long, val parameterIndex: Int = -1,
                                       val extraMapping: MatExpr = Mat.eye(3, 3, opencv_core.CV_64F)) {
        val trainable: Boolean = (parameterIndex >= 0)

        /*
         * If a ModelDescriptor gets referenced,
         * all model indices at which the corresponding ModelDescriptor references it will be stored in "associates".
         *
         * Note that this list will be empty for those non-trainable models,
         * as they can not be referenced
         *
         */
        val associates = mutableListOf<Long>()
    }


    private class Parameters(val numberOfImages: Int,
                             val modelDescriptors: Array<ModelDescriptor?>,
                             val correspondenceInfoList: List<Triple<Correspondence, Correspondence, Int>>,
                             val correspondenceMap: Map<Long, HashSet<Int>>,
                             val numberOfTrainable: Int,
                                  ) {


        // rows corresponds to all trainable parameters
        val cameraParams: Mat = Mat(  numberOfTrainable * (NUM_PARAMS_PER_INTRINSIC + NUM_PARAMS_PER_ROTATION) , 1, opencv_core.CV_64F)




        private val sliceCamParams = { start: Int, end: Int ->
            cameraParams.rowRange(start, end)
        }

        fun <INDEX: Number> getIntrinsicParams(which: INDEX, slicer: (Int, Int) -> Mat = sliceCamParams): Mat {
            val start = which.toInt() * NUM_PARAMS_PER_INTRINSIC

            return slicer(
                start,
                start + NUM_PARAMS_PER_INTRINSIC
            )
        }

        fun <INDEX: Number> getRotationParams(which: INDEX, slicer: (Int, Int) -> Mat = sliceCamParams): Mat {


            val start = numberOfTrainable * NUM_PARAMS_PER_INTRINSIC + which.toInt() * NUM_PARAMS_PER_ROTATION

            return slicer(
                start,
                start + NUM_PARAMS_PER_ROTATION
            )
        }
    }


    companion object{
        private const val NUM_PARAMS_PER_ROTATION = 3
        private const val NUM_PARAMS_PER_INTRINSIC = 1
        private const val DERIVATIVE_STEP = 1e-5


    }


    private var parametersProvider: Parameters? = null

    private val parameters: Parameters
        get() { return parametersProvider!! }




    private fun buildPinholeModel(index: Int): SimplifiedPinholeModel {



        val R = Mat()
        val invK = Mat.eye(3, 3, opencv_core.CV_64F).asMat()

        val modelDescriptor = parameters.modelDescriptors[index]!!
        val logF = if (modelDescriptor.trainable) {
            val rodrigues = parameters.getRotationParams(modelDescriptor.parameterIndex)
            opencv_calib3d.Rodrigues(rodrigues, R)

            parameters.getIntrinsicParams(modelDescriptor.parameterIndex).ptr().double
        }
        else {
            val referencedModelDescriptor = parameters.modelDescriptors[modelDescriptor.referenceIndex.toInt()]!!
            val referenceRodrigues = parameters.getRotationParams(referencedModelDescriptor.parameterIndex)
            opencv_calib3d.Rodrigues(referenceRodrigues, R)

            parameters.getIntrinsicParams(referencedModelDescriptor.parameterIndex).ptr().double
        }
        R.put(opencv_core.multiply(R, modelDescriptor.extraMapping))

        invK.ptr(0, 0).putDouble(exp(-logF))
        invK.ptr(1, 1).putDouble(exp(-logF))

        return SimplifiedPinholeModel(invK, R)
    }

    private fun buildPinholeModels(): Array<SimplifiedPinholeModel> {

        val models = Array(parameters.numberOfImages) { i ->
            buildPinholeModel(i)
        }

        return models
    }


    private suspend fun calculateError(worker: CVAsyncWorker, pinholeModels: Array<SimplifiedPinholeModel>, error: Mat) {
        for(correspondenceInfo in parameters.correspondenceInfoList) {
            worker.add {
                calculateError(pinholeModels, error, correspondenceInfo)
            }
        }
        worker.awaitAll()
    }

    private fun calculateError(pinholeModels: Array<SimplifiedPinholeModel>, error: Mat, correspondenceInfo: Triple<Correspondence, Correspondence, Int>) {
            
        val from = correspondenceInfo.first.index.toInt()
        val to = correspondenceInfo.second.index.toInt()


        val invFromK = pinholeModels[from].invK
        val invToK = pinholeModels[to].invK

        val fromR = pinholeModels[from].R
        val toR = pinholeModels[to].R


        val numberOfRows = correspondenceInfo.first.keypoints.rows()
        val numberOfColumns = correspondenceInfo.first.keypoints.cols()

        val tooSmallLength =
            Mat(numberOfRows, 1, correspondenceInfo.first.keypoints.type(), Scalar.all(1e-10))
        val fromLengthSquared = Mat()
        val fromLength = Mat()
        val toLengthSquared = Mat()
        val toLength = Mat()

        /* The objective is to calculate the (2.0 * cos_distance) between
         * p1' = p1 * invFromK^T * fromR^T
         * and
         * p2' = p2 * invToK^T * toR^T
         * , which can be obtained by norm( p1' / norm(p1') - p2' / norm(p2') )^2
         *
         */

        val fromP = opencv_core.multiply(correspondenceInfo.first.keypoints, opencv_core.multiply(invFromK.t(), fromR.t()))
        val toP = opencv_core.multiply(correspondenceInfo.second.keypoints, opencv_core.multiply(invToK.t(), toR.t()))




        opencv_core.reduce(
            fromP.mul(fromP).asMat(),
            fromLengthSquared,
            1,
            opencv_core.REDUCE_SUM
        )
        opencv_core.reduce(toP.mul(toP).asMat(), toLengthSquared, 1, opencv_core.REDUCE_SUM)

        opencv_core.sqrt(fromLengthSquared, fromLength)
        opencv_core.sqrt(toLengthSquared, toLength)


        val mask = Mat()
        val mask64 = Mat()


        opencv_core.compare(fromLength, tooSmallLength, mask, opencv_core.CMP_GT)
        mask.convertTo(mask64, opencv_core.CV_64F)
        mask64.put(opencv_core.divide(mask64, 255.0))

        val fromScale = opencv_core.divide(
            mask64,
            opencv_core.add(
                fromLength.mul(mask64),
                opencv_core.subtract(Mat.ones(mask64.size(), mask64.type()), mask64)
            )
        )


        opencv_core.compare(toLength, tooSmallLength, mask, opencv_core.CMP_GT)
        mask.convertTo(mask64, opencv_core.CV_64F)
        mask64.put(opencv_core.divide(mask64, 255.0))

        val toScale = opencv_core.divide(
            mask64,
            opencv_core.add(
                toLength.mul(mask64),
                opencv_core.subtract(Mat.ones(mask64.size(), mask64.type()), mask64)
            )
        )


        val destination = error.rowRange(
            correspondenceInfo.third,
            (correspondenceInfo.third + numberOfRows)
        ).reshape(1, numberOfRows)

        //actually, the variable "cosDistance" stores the value of cosine distances * 2
        var cosDistance = Mat.zeros(destination.size(), destination.type())

        for (columnIndex in 0 until numberOfColumns) {
            val diff = opencv_core.subtract(
                fromP.col(columnIndex).mul(fromScale),
                toP.col(columnIndex).mul(toScale)
            )
            cosDistance = opencv_core.add(cosDistance, diff.mul(diff))
        }

        destination.put(cosDistance)






        
    }

    private fun calculatePartialDerivative(variable: BytePointer, pinholeModels: Array<SimplifiedPinholeModel>, modelIndex: Int, associatedModelIndices: List<Long>, precomputedError: Mat): MatExpr {
        val modelIndexAsLong = modelIndex.toLong()
        if (modelIndexAsLong !in parameters.correspondenceMap && associatedModelIndices.isEmpty()) {
            return Mat.zeros(precomputedError.size(), precomputedError.type())
        }

        val value = variable.double
        val modelBackup = mutableMapOf<Int, SimplifiedPinholeModel>()
        modelBackup[modelIndex] = pinholeModels[modelIndex]

        /*
         * Getting two copies of precomputedError and only modifying the copied versions,
         * so that not only is the original error retained,
         * but race conditions can be avoided
         */
        val error0 = precomputedError.clone()
        val error1 = precomputedError.clone()


        val correspondenceInfoIndices = HashSet<Int>()
        parameters.correspondenceMap[modelIndexAsLong]?.let { indexSet ->
            correspondenceInfoIndices.addAll(indexSet)
        }

        /*
         * By design, for each camera model there should be only one thread to modify its parameters at a time
         * so it is safe to update and rebuild them here
         */
        variable.putDouble(value - DERIVATIVE_STEP)
        pinholeModels[modelIndex] = buildPinholeModel(modelIndex)

        associatedModelIndices.forEach {
            parameters.correspondenceMap[it]?.run {
                correspondenceInfoIndices.addAll(this)

                val indexAsInt = it.toInt()
                modelBackup[indexAsInt] = pinholeModels[indexAsInt]
                pinholeModels[indexAsInt] = buildPinholeModel(indexAsInt)
            }

        }

        correspondenceInfoIndices.forEach {
            calculateError(pinholeModels, error0, parameters.correspondenceInfoList[it])
        }

        variable.putDouble(value + DERIVATIVE_STEP)
        modelBackup.keys.forEach {
            pinholeModels[it] = buildPinholeModel(it)
        }

        correspondenceInfoIndices.forEach {
            calculateError(pinholeModels, error1, parameters.correspondenceInfoList[it])
        }

        variable.putDouble(value)
        modelBackup.forEach { (modelIndex, model) ->
            pinholeModels[modelIndex] = model
        }

        return opencv_core.divide(opencv_core.subtract(error1, error0), 2 * DERIVATIVE_STEP)

    }

    private suspend fun calculateJacobian(worker: CVAsyncWorker, pinholeModels: Array<SimplifiedPinholeModel>, jac: Mat, precomputedError: Mat)  {

        for (i in 0 until parameters.numberOfImages)
        {
            worker.add {


                val modelDescriptor = parameters.modelDescriptors[i]!!
                if (modelDescriptor.trainable) {

                    val copiedModels = pinholeModels.clone()

                    val intrinsicParams = parameters.getIntrinsicParams(modelDescriptor.parameterIndex)
                    val jacSliceForIntrinsic = parameters.getIntrinsicParams(modelDescriptor.parameterIndex) { start, end ->
                        jac.colRange(start, end)
                    }
                    for (j in 0 until NUM_PARAMS_PER_INTRINSIC) {

                        /*
                         * Under this parameterisation, updates to intrinsic parameters of one model could affect others,
                         * thus a pre-compiled list of associatedModelIndices indicating which models would be affected
                         * is passed to calculatePartialDerivative(), to make sure all contributions to errors associated with the specific update are included
                         */
                        val derivative = calculatePartialDerivative(
                            intrinsicParams.ptr(j, 0),
                            copiedModels,
                            i,
                            modelDescriptor.associates,
                            precomputedError
                        )
                        jacSliceForIntrinsic.col(j).put(derivative)


                    }


                    val rotationParams = parameters.getRotationParams(modelDescriptor.parameterIndex)
                    val jacSliceForRotation = parameters.getRotationParams(modelDescriptor.parameterIndex) { start, end ->
                        jac.colRange(start, end)
                    }


                    for (j in 0 until NUM_PARAMS_PER_ROTATION) {
                        /*
                         * Under this parameterisation, updates to rotation parameters of one model could affect the rotation parameters of others;
                         * thus a pre-compiled list of associatedModelIndices indicating which models would be affected
                         * is passed to calculatePartialDerivative(), to make sure all contributions to errors associated with the specific update are included
                         */
                        val derivative = calculatePartialDerivative(
                            rotationParams.ptr(j, 0),
                            copiedModels,
                            i,
                            modelDescriptor.associates,
                            precomputedError
                        )
                        jacSliceForRotation.col(j).put(derivative)


                    }
                }
            }

        }

        worker.awaitAll()

    }



    suspend fun estimate(
        features: ImageFeaturesVector,
        pairwiseMatches: MatchesInfoVector,
        cameras: CameraParamsVector,
        poseList: List<MatExpr>,
        is360: Boolean,
        confidenceThreshold: Double,
        termCriteria: TermCriteria = TermCriteria(TermCriteria.EPS + TermCriteria.COUNT, 100, 1e-3),
        updateIteration: suspend (run: Int) -> Unit = {}
    ): Boolean {

        val modelDescriptors = Array<ModelDescriptor?>(poseList.size) { null }
        var numberOfTrainableRotations = 0

        var referenceIndex = 0L
        val endIndex = if (is360) {
            features.size() - 1
        }
        else {
            features.size() - 2
        }

        var numberOfCorrespondence = 0
        val correspondenceInfoList = ArrayList<Triple<Correspondence, Correspondence, Int>>((endIndex+1).toInt())

        val correspondenceMap = mutableMapOf<Long, HashSet<Int>>()



        for (from in 0..endIndex)
        {
            val fromFeatures = features[from]
            val to = (from+1)%features.size()
            val toFeatures = features[to]
            val matchesInfo = pairwiseMatches[from * features.size() + to]

            val fromCentre = Pair(fromFeatures.img_size().width() * 0.5, fromFeatures.img_size().height() * 0.5)
            val toCentre = Pair(toFeatures.img_size().width() * 0.5, toFeatures.img_size().height() * 0.5)

            /*
             * For those matched pairs with low confidence,
             * excluding them from contributing optimisation errors,
             * as optimising over those spurious matches could end up with a bad estimate of parameters
             *
             * Also when exclusion happens, the parameters of the second camera (with "to" index) are considered "fixed",
             * in a way that both of its intrinsic and rotation parameters are calculated based on its reference
             * rather than optimised freely
             */

            if (matchesInfo.confidence() < confidenceThreshold) {
                modelDescriptors[to.toInt()] = ModelDescriptor(referenceIndex)
            }
            else {

                modelDescriptors[to.toInt()] = ModelDescriptor(-1, numberOfTrainableRotations)
                numberOfTrainableRotations += 1
                referenceIndex = to



                for (modelIndex in arrayOf(from, to)) {
                    correspondenceMap.getOrPut(modelIndex) {
                        HashSet(2)
                    }.add(correspondenceInfoList.size)
                }

                // Setting up correspondences of keypoints

                val fromKeypoints = Mat.ones(matchesInfo.num_inliers(), 3, opencv_core.CV_64F).asMat()
                val toKeypoints = fromKeypoints.clone()

                val numberOfMatches = matchesInfo.matches().size()
                val inlierMask = matchesInfo.inliers_mask()
                var kpIndex = 0
                for (l in 0 until numberOfMatches) {
                    if (!inlierMask.getBool(l))
                        continue
                    val m = matchesInfo.matches()[l]
                    val fromP = fromFeatures.keypoints()[m.queryIdx().toLong()].pt()
                    val toP = toFeatures.keypoints()[m.trainIdx().toLong()].pt()


                    fromKeypoints.ptr(kpIndex, 0)
                        .putDouble(fromP.x().toDouble() - fromCentre.first)
                    fromKeypoints.ptr(kpIndex, 1)
                        .putDouble(fromP.y().toDouble() - fromCentre.second)

                    toKeypoints.ptr(kpIndex, 0)
                        .putDouble(toP.x().toDouble() - toCentre.first)
                    toKeypoints.ptr(kpIndex, 1)
                        .putDouble(toP.y().toDouble() - toCentre.second)




                    kpIndex += 1
                }

                correspondenceInfoList.add(
                    Triple(
                        Correspondence(from, fromKeypoints),
                        Correspondence(to, toKeypoints),
                        numberOfCorrespondence
                    )
                )

                numberOfCorrespondence += matchesInfo.num_inliers()
            }

        }


        /*
         * when the sequence does not cover 360 degrees, or it does cover but not containing any confident matches,
         * the parameters of the first camera (with index 0) model is forced to be trainable,
         * so that there is at least one available for others to reference
         */
        if (!is360 || numberOfTrainableRotations == 0) {
            modelDescriptors[0] = ModelDescriptor(-1, numberOfTrainableRotations)
            numberOfTrainableRotations += 1
        }





        parametersProvider = Parameters(
            features.size().toInt(),
            modelDescriptors,
            correspondenceInfoList,
            correspondenceMap,
            numberOfTrainableRotations
        )



        for( i in 0 until parameters.numberOfImages) {
            //val cameraIndex = (firstTrainableIndex+i) % features.size()
            val camera = cameras[i.toLong()]
            Log.d(context.resources.getString(R.string.app_name), "camera $i: focal = ${camera.focal()}")
            // focal lengths are expressed in natural logarithm, to constrain them to be non-negative
            //parameters.getIntrinsicParams(i).ptr().putDouble(ln(camera.focal()))



            val rotationDescriptor = parameters.modelDescriptors[i]!!
            if (rotationDescriptor.trainable) {
                parameters.getIntrinsicParams(rotationDescriptor.parameterIndex).ptr().putDouble(ln(camera.focal()))

                opencv_calib3d.Rodrigues(
                    poseList[i].asMat(),
                    parameters.getRotationParams(rotationDescriptor.parameterIndex)
                )
            }
            else {
                /*
                 * Since non-trainable parameters of a camera model are calculated based on its reference,
                 * we want references to be trainable (thus their parameters can be read out directly without further referencing);
                 * otherwise there might be an endless referencing cycle in cases where the given sequence covers 360 degrees
                 */

                referenceIndex = rotationDescriptor.referenceIndex
                var referencedRelativeRotation = parameters.modelDescriptors[referenceIndex.toInt()]!!
                while (!referencedRelativeRotation.trainable) {
                    referenceIndex = referencedRelativeRotation.referenceIndex
                    referencedRelativeRotation = modelDescriptors[referenceIndex.toInt()]!!

                }
                val transitionR = opencv_core.multiply(poseList[referenceIndex.toInt()].t(), poseList[i])
                parameters.modelDescriptors[i] = ModelDescriptor(referenceIndex, extraMapping = transitionR)
                referencedRelativeRotation.associates.add(i.toLong())

            }

        }

        modelDescriptors.forEachIndexed { index, descriptor ->
            Log.d("", "camera ${index}: ${if (descriptor!!.trainable) "trainable" else "referencing ${descriptor.referenceIndex}" }")
        }

        /*
         * If no correspondences available, since there is no error to optimise against,
         * the whole optimisation process is skipped
         */
        if (numberOfCorrespondence > 0) {
            //val lockMask = Mat(parameters.cameraParams.size(), opencv_core.CV_8U, Scalar.all(1.0) )

            val solver = CvLevMarq(
                parameters.cameraParams.rows(),
                numberOfCorrespondence,
                opencv_core.cvTermCriteria(termCriteria),
                false
            )
            //opencv_core.cvCopy(opencv_core.cvMat(parameters.cameraParamsMask), solver.mask())

            val asyncWorker = CVAsyncWorker(
                max(
                    parameters.numberOfImages,
                    parameters.correspondenceInfoList.size
                ), CoroutineScope(coroutineContext)
            )

            val error = Mat.zeros(numberOfCorrespondence, 1, opencv_core.CV_64F).asMat()
            val jacobian =
                Mat.zeros(numberOfCorrespondence, parameters.cameraParams.rows(), opencv_core.CV_64F).asMat()


            val wrappedJacobian = opencv_core.cvMat(jacobian)
            val wrappedError = opencv_core.cvMat(error)
            val wrappedParams = opencv_core.cvMat(parameters.cameraParams)

            opencv_core.cvCopy(wrappedParams, solver.param())


            var run = 0
            while (true) {

                updateIteration(run)

                val currentParam = CvMat()
                val currentJacbian = CvMat()
                val currentError = CvMat()
                val proceed = solver.update(currentParam, currentJacbian, currentError)


                Log.d(
                    context.resources.getString(R.string.app_name),
                    "run $run: the norm of param difference = ${
                        opencv_core.cvNorm(
                            solver.param(),
                            solver.prevParam()
                        )
                    }"
                )


                opencv_core.cvCopy(currentParam, wrappedParams)

                if (!proceed || solver.state() == CvLevMarq.DONE) {
                    Log.d(
                        context.resources.getString(R.string.app_name),
                        "solver has finished the optimisation"
                    )
                    break
                }

                val models = buildPinholeModels()

                if (!currentError.isNull || !currentJacbian.isNull)
                    calculateError(asyncWorker, models, error)

                if (!currentError.isNull) {

                    val errorNorm = opencv_core.cvNorm(wrappedError)
                    Log.d(
                        context.resources.getString(R.string.app_name),
                        "run $run: the norm of error = $errorNorm"
                    )

                    /*
                    if (errorNorm <= termCriteria.epsilon()) {
                        Log.d(
                            context.resources.getString(R.string.app_name),
                            "errorNorm <= the designated epsilon criterion ${termCriteria.epsilon()}"
                        )
                        break
                    }

                     */

                    opencv_core.cvCopy(wrappedError, currentError)

                }

                if (!currentJacbian.isNull) {
                    calculateJacobian(asyncWorker, models, jacobian, error)
                    opencv_core.cvCopy(wrappedJacobian, currentJacbian)

                }

                run += 1
            }
        }

        // Check if all camera parameters are valid
        var ok = true
        for (i in 0 until parameters.cameraParams.rows())
        {
            if (!parameters.cameraParams.ptr(i,0).double.isFinite())
            {
                ok = false
                break
            }
        }
        if (ok) {

            for (i in 0 until parameters.numberOfImages) {

                val camera = cameras[i.toLong()]
                val R = camera.R()
                val modelDescriptor = parameters.modelDescriptors[i]!!



                if (modelDescriptor.trainable) {
                    camera.focal(exp(parameters.getIntrinsicParams(modelDescriptor.parameterIndex).ptr().double))
                    opencv_calib3d.Rodrigues(parameters.getRotationParams(modelDescriptor.parameterIndex), R)


                } else {
                    val referencedModelDescriptor = parameters.modelDescriptors[modelDescriptor.referenceIndex.toInt()]!!
                    camera.focal(exp(parameters.getIntrinsicParams(referencedModelDescriptor.parameterIndex).ptr().double))
                    opencv_calib3d.Rodrigues(
                        parameters.getRotationParams(referencedModelDescriptor.parameterIndex),
                        R
                    )
                }

                R.put(opencv_core.multiply(R, modelDescriptor.extraMapping))


                /*
                Log.d(
                    context.resources.getString(nothingstopsme.panorama.R.string.app_name),
                    "after optimisation(camera $i): focal = ${camera.focal()}, rotation matrix =\n" +
                            "${camera.R().ptr(0, 0).double}, ${
                                camera.R().ptr(0, 1).double
                            }, ${camera.R().ptr(0, 2).double}\n" +
                            "${camera.R().ptr(1, 0).double}, ${
                                camera.R().ptr(1, 1).double
                            }, ${camera.R().ptr(1, 2).double}\n" +
                            "${camera.R().ptr(2, 0).double}, ${
                                camera.R().ptr(2, 1).double
                            }, ${camera.R().ptr(2, 2).double}"
                )

                */


            }



            /*
            // Aligning the centre of motions to the centre of the image sequence
            val spanTree = Graph()
            val spanTreeCenters = IntArray(features.size().toInt()){ -1 }
            opencv_stitching.findMaxSpanningTree(
                features.size().toInt(),
                pairwiseMatches,
                spanTree,
                spanTreeCenters
            )
            val invR = cameras[spanTreeCenters[0].toLong()].R().inv().asMat()

            for (i in 0 until features.size()) {
                cameras[i].R().put(opencv_core.multiply(invR, cameras[i].R()))
            }

            */



        }

        parametersProvider = null

        return ok
    }


}