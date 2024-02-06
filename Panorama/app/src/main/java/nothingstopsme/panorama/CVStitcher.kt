package nothingstopsme.panorama

import android.os.Environment
import android.util.Log
import kotlinx.coroutines.CoroutineScope


import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

import kotlinx.coroutines.withContext
import org.bytedeco.opencv.global.opencv_core
import org.bytedeco.opencv.global.opencv_imgcodecs
import org.bytedeco.opencv.global.opencv_imgproc
import org.bytedeco.opencv.global.opencv_stitching
import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.opencv_core.MatExpr
import org.bytedeco.opencv.opencv_core.MatVector
import org.bytedeco.opencv.opencv_core.PointVector
import org.bytedeco.opencv.opencv_core.Scalar
import org.bytedeco.opencv.opencv_core.Size
import org.bytedeco.opencv.opencv_core.SizeVector
import org.bytedeco.opencv.opencv_core.UMat
import org.bytedeco.opencv.opencv_core.UMatVector
import org.bytedeco.opencv.opencv_stitching.BestOf2NearestMatcher
import org.bytedeco.opencv.opencv_stitching.BundleAdjusterRay
import org.bytedeco.opencv.opencv_stitching.BlocksGainCompensator
import org.bytedeco.opencv.opencv_stitching.CameraParams
import org.bytedeco.opencv.opencv_stitching.CameraParamsVector
import org.bytedeco.opencv.opencv_stitching.GraphCutSeamFinder
import org.bytedeco.opencv.opencv_stitching.HomographyBasedEstimator
import org.bytedeco.opencv.opencv_stitching.ImageFeaturesVector
import org.bytedeco.opencv.opencv_stitching.MatchesInfoVector
import org.bytedeco.opencv.opencv_stitching.MultiBandBlender
import org.bytedeco.opencv.opencv_stitching.SphericalWarper
import org.bytedeco.opencv.opencv_xfeatures2d.SURF
import org.bytedeco.opencv.opencv_core.TermCriteria
import java.io.File
import java.lang.RuntimeException
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Locale
import kotlin.math.cos

import kotlin.math.min
import kotlin.math.round
import kotlin.math.sin
import kotlin.math.sqrt

class CVStitcher(private val context: android.content.Context) {
    companion object {
        private val dateTimeFormat = SimpleDateFormat("yyyy_MM_dd-HH_mm_ss", Locale.ENGLISH)

        init {
            opencv_core.setNumThreads((Runtime.getRuntime().availableProcessors() - 2).coerceAtLeast(1))
        }
    }

    open class MotionDescriptor {
        enum class Type {
            GENERAL,
            SEQUENTIAL_ROTATION
        }
        open val type = Type.GENERAL


    }


    class SequentialRotationDescriptor(val sequence: List<String>, angleStep: Double, val is360: Boolean) : MotionDescriptor() {
        override val type = Type.SEQUENTIAL_ROTATION

        private val rotationArrayList = ArrayList<MatExpr>(sequence.size).apply {
            val rotation = Mat.eye(3, 3, opencv_core.CV_64F).asMat()
            rotation.ptr(0, 0).putDouble(cos(angleStep))
            rotation.ptr(2, 0).putDouble(-sin(angleStep))
            rotation.ptr(0, 2).putDouble(sin(angleStep))
            rotation.ptr(2, 2).putDouble(cos(angleStep))

            for (i in sequence.indices) {
                if (i == 0) {
                    add(Mat.eye(3, 3, opencv_core.CV_64F))
                }
                else {
                    add(opencv_core.multiply(get(i-1), rotation))
                }
            }

        }

        val rotationList: List<MatExpr>
            get() = rotationArrayList


    }

    data class Result(val code: Code, val imgPath: String? = null) {

        enum class Code(val value: Int) {
            //UNKNOWN_ERROR(-255),
            INVALID_OUTPUT_PATH(-5),
            IO_ERROR(-4),
            DISCONTINUOUS_IMAGES(-3),
            //INVALID_INPUT_MASKS(-2),
            NOT_EXIST(-1),
            OK(0),
            NEED_MORE_IMGS(1),
            HOMOGRAPHY_EST_FAIL(2),
            CAMERA_PARAMS_ADJUST_FAIL(3);

        }

    }



    class ProgressReporter(private val coroutineScope: CoroutineScope, private val callback: suspend (Float) -> Unit) {

        private val mutex = Mutex()
        private var progress = 0.0f



        private var incrementStep = 0.0f
        private var stepCountdown = 0


        fun setupProgress(start: Float, offset: Float, steps:Int=1): ProgressReporter {

            if (start < 0.0f || start > 1.0f)
                throw IllegalArgumentException("Bad start value: $start")

            val incrementStep = offset / steps
            if (incrementStep <= 0.0f || incrementStep > 1.0f)
                throw IllegalArgumentException("Bad offset or steps value: offset = $offset, steps = $steps")

            progress = start
            this.incrementStep = incrementStep
            stepCountdown = steps

            return this
        }

        private suspend fun increment(stepsLowerbound: Int = 0) {
            mutex.withLock {
                if (stepCountdown <= stepsLowerbound)
                    return

                val newProgress = min(1.0f, progress+incrementStep)
                stepCountdown -= 1
                if (newProgress != progress) {
                    progress = newProgress
                    callback(progress)
                }

            }

        }

        private suspend fun complete() {
            mutex.withLock {
                if (stepCountdown <= 0)
                    return

                val newProgress = min(1.0f, progress+incrementStep*stepCountdown)
                stepCountdown = 0
                if (newProgress != progress) {
                    progress = newProgress
                    callback(progress)
                }

            }

        }

        suspend fun setProgress(progress:Float) = CVAsyncWorker.waitFor {
            if (progress < 0.0f || progress > 1.0f)
                throw IllegalArgumentException("Bad progress value: $progress")

            mutex.withLock {
                incrementStep = 0f
                stepCountdown = 0
                if (progress != this.progress) {
                    this.progress = progress
                    callback(this.progress)
                }
            }
        }


        fun launchIncrementingJob(stepsLowerbound: Int = 0) {
            coroutineScope.launch {
                increment(stepsLowerbound)
            }
        }

        fun launchCompletionJob() {
            coroutineScope.launch {
                complete()
            }
        }

    }


    private val preferences = UserPreferences(context)


    /*
     * Various compoments initialised with the same settings used in opencv_stitching.Stitcher.create(CV_Stitcher.PANORAMA)
     */
    private val seamFinder by lazy { GraphCutSeamFinder(GraphCutSeamFinder.COST_COLOR, 10000.0f, 1000.0f) }
    private val blender by lazy { MultiBandBlender(0, 5, opencv_core.CV_32F) }
    //private val blender by lazy { FeatherBlender() }

    private val featuresFinder by lazy { SURF.create() }
    //private val featuresFinder by lazy { ORB.create() }
    private val estimator by lazy { HomographyBasedEstimator() }
    //private val featuresMatcher by lazy { BestOf2NearestMatcher(false, 0.3f, 6, 6, 3.0) }

    private val warper by lazy { SphericalWarper() }
    private val exposureCompensator by lazy { BlocksGainCompensator() }



    private val registrationResolutionWeight = 0.6
    private val seamEstimationResolutionWeight = 0.1
    private val compositionResolutionWeight = 0.9




    private val waveCorrectKind = opencv_stitching.WAVE_CORRECT_HORIZ


    private suspend fun stitch(inputDir: File, descriptor: MotionDescriptor, output: UMat, outputMask: UMat, progressReporter: ProgressReporter): Result.Code = coroutineScope {

        val (inputFiles, filesDesignation) = when (descriptor) {
            is SequentialRotationDescriptor -> {
                val orderMap = descriptor.sequence.withIndex().associate { item -> Pair(item.value, item.index) }


                val allFiles = inputDir.listFiles { f -> f.isFile && f.name in orderMap }!!
                if (allFiles.size != orderMap.size)
                    return@coroutineScope Result.Code.NEED_MORE_IMGS

                allFiles.sortBy { f -> orderMap[f.name] }
                Pair(allFiles, true)
            }
            else ->
                Pair(inputDir.listFiles { f -> f.isFile }!!, false)
        }



        val numberOfInputFiles = inputFiles.size.toLong()
        if (numberOfInputFiles < 2)
            return@coroutineScope Result.Code.NEED_MORE_IMGS


        val fullImageSizes = SizeVector().apply { capacity<SizeVector>(numberOfInputFiles) }
        val featureSearchImages = UMatVector().apply { capacity<UMatVector>(numberOfInputFiles) }
        val features = ImageFeaturesVector().apply { capacity<ImageFeaturesVector>(numberOfInputFiles) }
        val seamEstimationImages = UMatVector().apply { capacity<UMatVector>(numberOfInputFiles) }

        val validFiles = ArrayList<File>(inputFiles.size)

        val pairwiseMatches = MatchesInfoVector()
        val cameraParams = CameraParamsVector()

        var registrationScale = -1.0
        var seamScale = -1.0

        val openCLEnabled = preferences.openCLEnabled.first()
        val confidenceThreshold = preferences.minMatchConfidence.first()
        Log.d(context.getString(R.string.app_name), "openCLEnabled = $openCLEnabled")
        Log.d(context.getString(R.string.app_name), "confidenceThreshold = $confidenceThreshold")

        opencv_core.setUseOpenCL(openCLEnabled)
        // This is a helper class for conducting async works involving calls to opencv
        val asyncWorker = CVAsyncWorker(inputFiles.size, this@coroutineScope)


        progressReporter.setupProgress(0.0f, 0.2f,inputFiles.size*2)


        /*
         * Loading and scaling each image one by one instead of in parallel via coroutines,
         * as it could take up a lot of memory to handle images of a higher resolution
         */
        inputFiles.forEach { file ->

            val img = opencv_imgcodecs.imread(file.absolutePath).getUMat(opencv_core.ACCESS_READ)


            progressReporter.launchIncrementingJob()

            if (img.empty()) {
                Log.d(context.resources.getString(R.string.app_name), "Failed to read the image: '${file.absolutePath}'")
                if (filesDesignation)
                    return@coroutineScope Result.Code.IO_ERROR
            } else {


                fullImageSizes.push_back(img.size())


                if (registrationScale <= 0.0) {
                    registrationScale = if (registrationResolutionWeight < 0)
                        1.0
                    else
                        min(1.0, sqrt(registrationResolutionWeight * 1e6 / fullImageSizes[0].area()))
                }

                if (seamScale <= 0.0) {
                    seamScale = min(1.0, sqrt(seamEstimationResolutionWeight * 1e6 / fullImageSizes[0].area()))
                }


                var scaledImg = if (registrationScale == 1.0) {
                    img
                } else {
                    val scaledImg = UMat()
                    opencv_imgproc.resize(
                        img,
                        scaledImg,
                        Size(),
                        registrationScale,
                        registrationScale,
                        opencv_imgproc.INTER_LINEAR_EXACT
                    )
                    scaledImg
                }
                featureSearchImages.push_back(scaledImg)

                scaledImg = UMat()
                opencv_imgproc.resize(
                    img,
                    scaledImg,
                    Size(),
                    seamScale,
                    seamScale,
                    opencv_imgproc.INTER_LINEAR_EXACT
                )
                seamEstimationImages.push_back(scaledImg)

                img.release()

                val featureImageIndex = (featureSearchImages.size()-1).toInt()
                validFiles.add(file)
                features.resize(features.size()+1)
                features[features.size()-1].img_idx(featureImageIndex)
            }

            progressReporter.launchIncrementingJob()



        }



        val seamRegistrationRatio = seamScale / registrationScale
        var numberOfValidFilesAsLong = validFiles.size.toLong()


        opencv_stitching.computeImageFeatures(featuresFinder, featureSearchImages, features)
        featureSearchImages.clear()


        progressReporter.setupProgress(0.2f,0.02f).launchCompletionJob()

        val pairingList = when(descriptor) {
            is SequentialRotationDescriptor -> {

                val end = if (descriptor.is360)
                    features.size().toInt() - 1
                else
                    features.size().toInt() - 2

                IntRange(0, end).map { index ->
                    Pair(
                        index, (index+1)%features.size().toInt()

                    )
                }
            }
            else -> listOf()
        }

        if (pairingList.isEmpty()) {
            val featuresMatcher = BestOf2NearestMatcher(false, 0.3f, 6, 6, 3.0)
            featuresMatcher.apply2(features, pairwiseMatches)
            featuresMatcher.collectGarbage()

            progressReporter.setupProgress(0.22f, 0.02f).launchCompletionJob()
        }
        else {
            progressReporter.setupProgress(0.22f, 0.02f, pairingList.size)

            // Doing matching manually according to the pairing relationships presented in pairingMap
            pairwiseMatches.resize(features.size() * features.size())
            val rngState = opencv_core.theRNG().state()

            pairingList.onEachIndexed { entryIndex, entry ->
                asyncWorker.add {
                    opencv_core.theRNG().state(rngState + entryIndex)
                    //cv::theRNG() = cv::RNG(rng.state + i); // force "stable" RNG seed for each processed pair

                    val featuresMatcher = BestOf2NearestMatcher(false, 0.3f, 6, 6, 3.0)

                    val from = entry.first
                    val to = entry.second
                    val pairIndex = from * features.size() + to
                    val dualPairIndex = to * features.size() + from

                    val matchesInfo = pairwiseMatches[pairIndex]


                    try {
                        featuresMatcher.apply(
                            features[from.toLong()],
                            features[to.toLong()],
                            matchesInfo
                        )
                    }
                    catch (e: RuntimeException) {
                        Log.d(context.resources.getString(R.string.app_name), e.toString())

                        /*
                         * When a default camera motion model is available,
                         * the stitching process continues whether featuresMatcher can finish the matching process successfully or not.
                         * (Note that noisy images or images with few/no valid feature points could easily lead to the failure of matching)
                         */
                        when (descriptor) {
                            is SequentialRotationDescriptor -> {
                                matchesInfo.num_inliers(0)
                                matchesInfo.confidence(0.0)
                                matchesInfo.matches().clear()
                                matchesInfo.H(Mat())
                            }
                            else -> throw e
                        }

                    }




                    matchesInfo.src_img_idx(from)
                    matchesInfo.dst_img_idx(to)

                    val dualMatchesInfo = pairwiseMatches[dualPairIndex]
                    dualMatchesInfo.put(matchesInfo)

                    dualMatchesInfo.src_img_idx(to)
                    dualMatchesInfo.dst_img_idx(from)
                    if (!matchesInfo.H().empty()) {
                        dualMatchesInfo.H(matchesInfo.H().inv().asMat())
                    }


                    val matches = dualMatchesInfo.matches()
                    for (j in 0 until matches.size()) {
                        val match = matches[j]
                        val temp = match.queryIdx()
                        match.queryIdx(match.trainIdx())
                        match.trainIdx(temp)
                    }

                    featuresMatcher.collectGarbage()

                    progressReporter.launchIncrementingJob()
                }


            }






            asyncWorker.awaitAll()



        }




        val focals = DoubleArray(features.size().toInt())

        when (descriptor) {
            is SequentialRotationDescriptor -> {
                val termCriteria = TermCriteria(TermCriteria.EPS + TermCriteria.COUNT, 100, 1e-3)
                progressReporter.setupProgress(0.24f, 0.06f, termCriteria.maxCount()).launchCompletionJob()



                opencv_stitching.estimateFocal(features, pairwiseMatches, focals)

                cameraParams.resize(features.size())

                for (i in 0 until features.size()) {
                    cameraParams[i].focal(focals[i.toInt()])

                    cameraParams[i].ppx(0.5 * features[i].img_size().width())
                    cameraParams[i].ppy(0.5 * features[i].img_size().height())

                }


                val bundleAdjuster = SequentialRotationBundleAdjuster(context)

                if(!bundleAdjuster.estimate(features, pairwiseMatches, cameraParams, descriptor.rotationList, descriptor.is360, confidenceThreshold.toDouble(), termCriteria) {
                        progressReporter.launchIncrementingJob(termCriteria.maxCount()-1)
                        Log.d("", "optimistaion run $it")
                    })
                    return@coroutineScope Result.Code.CAMERA_PARAMS_ADJUST_FAIL

                progressReporter.launchCompletionJob()
            }
            else -> {
                val indices = opencv_stitching.leaveBiggestComponent(
                    features,
                    pairwiseMatches,
                    confidenceThreshold
                )

                if (indices.limit() != numberOfValidFilesAsLong)
                    return@coroutineScope Result.Code.DISCONTINUOUS_IMAGES

                progressReporter.setupProgress(0.24f, 0.02f).launchCompletionJob()

                if (!estimator.apply(features, pairwiseMatches, cameraParams))
                    return@coroutineScope Result.Code.HOMOGRAPHY_EST_FAIL

                progressReporter.setupProgress(0.26f, 0.02f).launchCompletionJob()

                val bundleAdjuster = BundleAdjusterRay()
                bundleAdjuster.setConfThresh(confidenceThreshold.toDouble())
                if (!bundleAdjuster.apply(features, pairwiseMatches, cameraParams))
                    return@coroutineScope Result.Code.CAMERA_PARAMS_ADJUST_FAIL

                progressReporter.setupProgress(0.28f, 0.02f).launchCompletionJob()
            }
        }


        for (i in 0 until cameraParams.size())
        {
            val convertedR = Mat()
            cameraParams[i].R().convertTo(convertedR, opencv_core.CV_32F)
            cameraParams[i].R(convertedR)

            focals[i.toInt()] = cameraParams[i].focal()
        }

        focals.sort()

        val warpedImageScale = if (focals.size % 2 == 1)
            focals[focals.size / 2]
        else
            (focals[focals.size / 2 - 1] + focals[focals.size / 2]) * 0.5


        features.clear()



        val rotationMatrices = MatVector().apply { capacity<MatVector>(cameraParams.size()) }
        for (i in 0 until cameraParams.size())
            rotationMatrices.push_back(cameraParams[i].R())

        opencv_stitching.waveCorrect(rotationMatrices, waveCorrectKind)
        rotationMatrices.clear()


        if (descriptor is SequentialRotationDescriptor) {
            /*
             * It looks like opencv_stitching.waveCorrect() could also alter the motion centre,
             * so the centre adjustment is done here (after opencv_stitching.waveCorrect()) instead of inside SequentialRotationBundleAdjuster
             */

            val centre = cameraParams.size() / 2
            Log.d(context.resources.getString(R.string.app_name),
                "centring the stitch result at camera $centre")
            /*
             * Because the R of the camera at the centre will be changed in the following loop,
             * invR has to be a copy of Mat (rather than MatExpr, which references back to the original camera parameter)
             * to avoid race conditions
             */
            val invR = cameraParams[centre].R().inv().asMat()
            for (i in 0 until cameraParams.size()) {
                cameraParams[i].R().put(opencv_core.multiply(invR, cameraParams[i].R()))
            }
        }

        progressReporter.setupProgress(0.3f, 0.02f).launchCompletionJob()

        // Generating warped images and masks used in seam finding

        val corners = PointVector(numberOfValidFilesAsLong)
        val warpedMasks = UMatVector(numberOfValidFilesAsLong)
        val warpedImages = UMatVector(numberOfValidFilesAsLong)



        progressReporter.setupProgress(0.32f, 0.1f, validFiles.size)

        for (i in 0 until numberOfValidFilesAsLong)
        {
            asyncWorker.add {
                val warpingWorker = warper.create((warpedImageScale * seamRegistrationRatio).toFloat())

                val mask = UMat(seamEstimationImages[i].size(), opencv_core.CV_8U, Scalar.all(255.0))

                /*
                 * Adjusting intrinsic and rotation matrix to the scale where seam finding performs.
                 * Note that the adjusted values can not overwrite the original ones,
                 * as they are still needed in the composition process later
                 */
                val params = CameraParams(cameraParams[i])
                val R = params.R().getUMat(opencv_core.ACCESS_READ)
                val K = UMat()


                params.ppx(params.ppx() * seamRegistrationRatio)
                params.ppy(params.ppy() * seamRegistrationRatio)
                params.focal(params.focal() * seamRegistrationRatio)

                params.K().convertTo(K, opencv_core.CV_32F)

                corners.put(
                    i,
                    warpingWorker.warp(
                        seamEstimationImages[i],
                        K,
                        R,
                        opencv_imgproc.INTER_LINEAR,
                        opencv_core.BORDER_REFLECT,
                        //opencv_core.BORDER_CONSTANT,
                        warpedImages[i]
                    )
                )
                //warpedImageSizes.put(i, warpedImages[i].size())


                warpingWorker.warp(
                    mask,
                    K,
                    R,
                    opencv_imgproc.INTER_NEAREST,
                    opencv_core.BORDER_CONSTANT,
                    warpedMasks[i]
                )

                progressReporter.launchIncrementingJob()


            }
        }
        asyncWorker.awaitAll()
        seamEstimationImages.clear()

        if (descriptor is SequentialRotationDescriptor && descriptor.is360) {

            val areas = IntArray(warpedImages.size().toInt())
            for (i in 0 until warpedImages.size()) {
                areas[i.toInt()] = warpedImages[i].size().area()
            }
            areas.sort()

            val mediumArea = if (areas.size % 2 == 0) {
                (areas[areas.size / 2 - 1] + areas[areas.size / 2]) / 2
            } else
                areas[areas.size / 2]

            Log.d(
                context.resources.getString(R.string.app_name),
                "mediumArea after warping '${mediumArea}'"
            )

            val problematicIndices = ArrayList<Long>(warpedImages.size().toInt())
            for (i in 0 until warpedImages.size()) {
                /*
                 * For outlier images with a irregularly large/small size,
                 * checking if they are involved in matches with high confidence;
                 * If so, since their camera model should not be too wrong,
                 * their warped results are considered valid even if there exists some irregularity;
                 * otherwise, throwing them away
                 */
                if (warpedImages[i].size().area() * 2 < mediumArea || warpedImages[i].size().area() > mediumArea * 2) {

                    val previous = (i - 1 + warpedImages.size()) % warpedImages.size()
                    val next = (i + 1) % warpedImages.size()

                    if (pairwiseMatches[i * warpedImages.size() + previous].confidence() < confidenceThreshold || pairwiseMatches[i * warpedImages.size() + next].confidence() < confidenceThreshold) {
                        problematicIndices.add(i)
                    }

                }

            }

            if (problematicIndices.isNotEmpty()) {

                for (i in 0 until problematicIndices.size) {
                    if (i > 0 && (problematicIndices[i] - problematicIndices[i-1]) != 1L) {
                        return@coroutineScope Result.Code.DISCONTINUOUS_IMAGES
                    }

                    Log.d(context.resources.getString(R.string.app_name), "throwing image ${problematicIndices[i]} away since its warped version looks wrong")
                }

                numberOfValidFilesAsLong -= problematicIndices.size
                problematicIndices.asReversed().forEach {
                    validFiles.removeAt(it.toInt())
                }


                for (index in 0 until numberOfValidFilesAsLong - problematicIndices.first()) {
                    fullImageSizes.put(index+problematicIndices.first(), fullImageSizes[index+problematicIndices.last()+1])
                    cameraParams.put(index+problematicIndices.first(), cameraParams[index+problematicIndices.last()+1])

                    corners.put(index+problematicIndices.first(), corners[index+problematicIndices.last()+1])
                    warpedImages.put(index+problematicIndices.first(), warpedImages[index+problematicIndices.last()+1])
                    warpedMasks.put(index+problematicIndices.first(), warpedMasks[index+problematicIndices.last()+1])


                }

                fullImageSizes.resize(numberOfValidFilesAsLong)
                cameraParams.resize(numberOfValidFilesAsLong)

                corners.resize(numberOfValidFilesAsLong)
                warpedImages.resize(numberOfValidFilesAsLong)
                warpedMasks.resize(numberOfValidFilesAsLong)
            }
        }

        pairwiseMatches.clear()

        // Compensating exposure before finding seams
        val warpedImagesAsFloat = UMatVector(warpedImages.size())
        exposureCompensator.feed(corners, warpedImages, warpedMasks)

        progressReporter.setupProgress(0.42f, 0.02f).launchCompletionJob()


        progressReporter.setupProgress(0.44f, 0.1f, warpedImages.size().toInt())
        for (i in 0 until warpedImages.size()) {
            asyncWorker.add {

                exposureCompensator.apply(i.toInt(), corners[i], warpedImages[i], warpedMasks[i])
                warpedImages[i].convertTo(warpedImagesAsFloat[i], opencv_core.CV_32F)

                progressReporter.launchIncrementingJob()

            }
        }
        asyncWorker.awaitAll()


        seamFinder.find(warpedImagesAsFloat, corners, warpedMasks)

        progressReporter.setupProgress(0.54f, 0.02f).launchCompletionJob()

        // Releasing unused memory
        warpedImagesAsFloat.clear()
        warpedImages.clear()
        warpedImages.resize(numberOfValidFilesAsLong)


        // Compositing

        val compositionScale = if (compositionResolutionWeight <= 0)
            1.0
        else
            min(1.0, sqrt(compositionResolutionWeight * 1e6 / fullImageSizes[0].area()))

        val compositionRegistrationRatio = compositionScale / registrationScale

        val warpedImageSizes = SizeVector(numberOfValidFilesAsLong)

        progressReporter.setupProgress(0.56f, 0.1f, validFiles.size)
        for (i in 0 until numberOfValidFilesAsLong)
        {
            asyncWorker.add {

                val warpingWorker = warper.create((warpedImageScale * compositionRegistrationRatio).toFloat())
                // Updating the intrinsic parameters
                cameraParams[i].ppx(cameraParams[i].ppx() * compositionRegistrationRatio)
                cameraParams[i].ppy(cameraParams[i].ppy() * compositionRegistrationRatio)
                cameraParams[i].focal(cameraParams[i].focal() * compositionRegistrationRatio)

                // Updating the size and corners to meet the composition scale
                val sz = fullImageSizes[i]
                //if (abs(compositionScale - 1) > 1e-1) {
                sz.width(round(sz.width().toDouble() * compositionScale).toInt())
                sz.height(round(sz.height().toDouble() * compositionScale).toInt())
                //}


                val K = UMat()
                cameraParams[i].K().convertTo(K, opencv_core.CV_32F)
                val R = cameraParams[i].R().getUMat(opencv_core.ACCESS_READ)
                val roi = warpingWorker.warpRoi(sz, K, R)

                corners.put(i, roi.tl())
                warpedImageSizes.put(i, roi.size())

                progressReporter.launchIncrementingJob()

            }
        }

        asyncWorker.awaitAll()



        blender.prepare(corners, warpedImageSizes)



        progressReporter.setupProgress(0.66f, 0.02f).launchCompletionJob()

        val warpedImage = UMat()
        val warpedMask = UMat()
        val compositionImage = UMat()
        val dilatedMask = UMat()
        val warpingWorker = warper.create((warpedImageScale * compositionRegistrationRatio).toFloat())


        progressReporter.setupProgress(0.68f, 0.3f, validFiles.size*3)

        /*
         * Blending each image one by one instead of in parallel via coroutines,
         * as it could take up a lot of memory to handle images of a higher resolution
         */
        for (i in 0 until numberOfValidFilesAsLong)
        {
            // Loading the i-th image from the corresponding file, and resizing it to the composition scale
            val path = validFiles[i.toInt()].absolutePath
            val inputImage = opencv_imgcodecs.imread(path).getUMat(opencv_core.ACCESS_READ)


            if (inputImage.empty()) {
                Log.d(
                    context.resources.getString(R.string.app_name),
                    "Failed to read the image during blending: '${path}'"
                )
                warpedMasks[i].release()
                continue
            }


            //if (abs(compositionScale - 1) > 1e-1) {
            if (compositionScale != 1.0) {
                opencv_imgproc.resize(
                    inputImage,
                    compositionImage,
                    Size(),
                    compositionScale,
                    compositionScale,
                    opencv_imgproc.INTER_LINEAR_EXACT
                )

            } else
                compositionImage.put(inputImage)

            inputImage.release()

            progressReporter.launchIncrementingJob()


            val imageSize = compositionImage.size()
            val K = UMat()
            cameraParams[i].K().convertTo(K, opencv_core.CV_32F)
            val R = cameraParams[i].R().getUMat(opencv_core.ACCESS_READ)

            // Warping the image to generate the result the blender needs
            warpingWorker.warp(
                compositionImage,
                K,
                R,
                opencv_imgproc.INTER_LINEAR,
                opencv_core.BORDER_REFLECT,
                //opencv_core.BORDER_CONSTANT,
                warpedImage
            )
            compositionImage.release()




            // Generate the warped mask corresponding to the warped image
            warpingWorker.warp(
                UMat(
                    imageSize,
                    opencv_core.CV_8U,
                    Scalar.all(255.0)
                ), K, R, opencv_imgproc.INTER_NEAREST, opencv_core.BORDER_CONSTANT, warpedMask
            )


            progressReporter.launchIncrementingJob()


            // Compensating exposure
            exposureCompensator.apply(i.toInt(), corners[i], warpedImage, warpedMask)


            warpedImage.convertTo(compositionImage, opencv_core.CV_16S)
            warpedImage.put(compositionImage)
            compositionImage.release()



            // Making sure seam mask has the same size as the one of warpedMask
            opencv_imgproc.dilate(warpedMasks[i], dilatedMask, UMat())
            opencv_imgproc.resize(
                dilatedMask,
                warpedMasks[i],
                warpedMask.size(),
                0.0,
                0.0,
                opencv_imgproc.INTER_LINEAR_EXACT
            )
            dilatedMask.release()
            opencv_core.bitwise_and(warpedMasks[i], warpedMask, warpedMask)
            warpedMasks[i].release()


            /*
            val outputDirPath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).absolutePath + "/" + context.resources.getString(
                nothingstopsme.panorama.R.string.app_name)
            val outputPath = "${outputDirPath}/dump/${validFiles[i.toInt()].name}"

            opencv_imgcodecs.imwrite(outputPath, warpedImage)


            Log.d("", "feed $i: $path, corner = (${corners[i].x()}, ${corners[i].y()})")

             */

            blender.feed(warpedImage, warpedMask, corners[i])



            warpedImage.release()
            warpedMask.release()


            progressReporter.launchIncrementingJob()


        }





        val result = UMat()
        blender.blend(result, outputMask)


        progressReporter.setupProgress(0.98f, 0.02f).launchCompletionJob()

        // Preliminary result is in CV_16SC3 format, but all values are in [0,255] range,
        // so convert it to avoid user confusing
        result.convertTo(output, opencv_core.CV_8U)

        result.release()

        return@coroutineScope Result.Code.OK
    }

    suspend fun stitch(inputDirPath: String, outputDirPath: String, descriptor: MotionDescriptor, progressCallback: suspend (Float) -> Unit): Result = withContext(Dispatchers.IO) {
        val inputDir = File(inputDirPath)

        if (!inputDir.isDirectory)
            return@withContext Result(Result.Code.NOT_EXIST)

        val outputDir = File(outputDirPath)

        if (outputDir.exists()) {
            if (!outputDir.isDirectory)
                return@withContext Result(Result.Code.INVALID_OUTPUT_PATH)
        }
        else
            outputDir.mkdirs()

        val panorama = UMat()
        val panoramaMask = UMat()


        val progressReporter = ProgressReporter(this, progressCallback)
        val resultCode = stitch(inputDir, descriptor, panorama, panoramaMask, progressReporter)
        Log.d(context.resources.getString(R.string.app_name), "stitching result = $resultCode")

        return@withContext if (resultCode == Result.Code.OK) {
            val now = dateTimeFormat.format(Calendar.getInstance().time)
            val outputPath = "${outputDirPath}/${now}.jpg"

            opencv_imgcodecs.imwrite(outputPath, panorama)

            panorama.release()
            panoramaMask.release()


            progressReporter.setProgress(1.0f)


            Result(resultCode, outputPath)

        }
        else
            Result(resultCode)









    }

}