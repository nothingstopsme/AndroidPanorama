package nothingstopsme.panorama

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorManager
import android.hardware.camera2.CaptureRequest
import android.os.Bundle
import android.os.Environment
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraControl
import androidx.camera.core.CameraUnavailableException
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import nothingstopsme.panorama.ui.theme.PanoramaTheme
import java.io.File
import java.util.concurrent.ExecutionException
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine
import kotlin.math.PI


class MainActivity : ComponentActivity() {



    companion object {

        const val NUM_OF_ANGLE_STEPS = 12
        const val HORIZONTAL_ANGLE_STEP_SIZE = 360.0f / NUM_OF_ANGLE_STEPS
        //const val VERTICAL_ANGLE_STEP_SIZE = 30.0f
    }

    private val tempPath by lazy {
        Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).absolutePath + "/" + resources.getString(
            R.string.app_name
        ) + "/temp"
    }
    private val panoramaOutputPath by lazy {
        Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).absolutePath + "/" + resources.getString(
            R.string.app_name
        ) + "/output"
    }

    //private val backgroundExecutor = Executors.newCachedThreadPool()

    private lateinit var cameraPreviewView: PreviewView

    private lateinit var sensorManager: SensorManager


    private val stitcher by lazy {
        CVStitcher(this)
    }



    private val deviceOrientationMonitor = DeviceOrientationMonitor()

    private val viewModel by viewModels<MainViewModel>()



    @ExperimentalCamera2Interop
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)





        val permissionsRequired = arrayOf(
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.CAMERA)


        for(permission in permissionsRequired)
        {
            if(ContextCompat.checkSelfPermission(baseContext, permission) != PackageManager.PERMISSION_GRANTED)
            {
                val requestPermissionLauncher = registerForActivityResult(
                    ActivityResultContracts.RequestMultiplePermissions()
                ) { isGrantedMap: Map<String, Boolean> ->
                    for ((_, isGranted) in isGrantedMap) {
                        if (!isGranted){
                            finish()
                            return@registerForActivityResult
                        }
                    }
                    initialise()
                }
                requestPermissionLauncher.launch(permissionsRequired)
                return
            }
        }

        initialise()
    }

    @ExperimentalCamera2Interop
    override fun onResume() {
        super.onResume()

        //var readyFlag = false
        if (viewModel.angleStepIndex.value == 0) {
            //readyFlag = true
            if (this::sensorManager.isInitialized) {

                val rotationSensor =
                    sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)


                if (rotationSensor != null) {
                    sensorManager.registerListener(
                        deviceOrientationMonitor,
                        rotationSensor,
                        SensorManager.SENSOR_DELAY_NORMAL
                    )

                    // hooking up the sensor update callback a bit later to avoid inaccurate sensor data sent to
                    // a listener who has just been registered
                    this.lifecycleScope.launch {
                        delay(1000)
                        deviceOrientationMonitor.onUpdate {
                            viewModel.updateTargetOffset(this)
                        }
                    }


                } else {
                    viewModel.popupMessage.value =
                        PopupMessage(getString(R.string.rotation_sensor_unavailable))
                    //readyFlag = false
                    return
                }

            }
        }


        preparePreview {
            // doing nothing
        }

        /*
        if (this::cameraPreviewView.isInitialized) {
            //viewModel.cameraPreview.setSurfaceProvider(cameraPreviewView.surfaceProvider)
            viewModel.updateCameraPreviewSurfaceProvider(cameraPreviewView.surfaceProvider)
        }
        *
         */



        viewModel.readyFlag.value = true

    }

    override fun onPause() {

        if (viewModel.angleStepIndex.value == 0) {
            if (this::sensorManager.isInitialized) {
                sensorManager.unregisterListener(deviceOrientationMonitor)
                deviceOrientationMonitor.onUpdate(null)
            }

            viewModel.releaseCamera()
        }
        viewModel.readyFlag.value = false
        super.onPause()
    }



    @ExperimentalCamera2Interop
    private fun initialise()
    {

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        //rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
        //sensorManager.registerListener(deviceOrientationMonitor, rotationSensor, SensorManager.SENSOR_DELAY_NORMAL)













        setContent {
            PanoramaTheme(darkTheme = true) {
                Navigation(main = {
                    MainScreen(
                        navController = it,
                        popupMessage = viewModel.popupMessage,
                        readyFlag = viewModel.readyFlag,
                        targetOffset = viewModel.targetOffset,
                        angleStepIndex = viewModel.angleStepIndex,
                        preparePreview = this@MainActivity::preparePreview,
                        preparePictureTaking = this@MainActivity::preparePictureTaking,
                        takePicture = this@MainActivity::takePicture,
                        stitch = this@MainActivity::doStitching,
                        reset = this@MainActivity::reset
                    )
                },
                    settings = {
                        SettingsScreen(navController = it, UserPreferences(this))
                    }
                )

            }
        }

    }


    @ExperimentalCamera2Interop
    private suspend fun reset() {




        withContext(Dispatchers.IO) {
            val dir =
                File(tempPath)
            if (dir.exists()) {
                dir.listFiles()?.forEach {
                    it.delete()
                }
            } else
                dir.mkdirs()

        }



        withContext(Dispatchers.Main) {

            viewModel.reset()

            viewModel.camera?.also {
                lockExposureAndWhiteBalance(false)

            }
        }


    }

    private fun doStitching(updateProgress: suspend (Float) -> Unit, onFinished: suspend () -> Unit) {



        val inputPath = tempPath
        //val inputPath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).absolutePath + "/" + getString(R.string.app_name) + "/temp_copy2"

        //val is360 = true
        val is360 = viewModel.angleStepIndex.value == NUM_OF_ANGLE_STEPS
        val sequence = mutableListOf<String>()


        for (s in 0 until viewModel.angleStepIndex.value) {
            sequence.add(generateTempImageName(s))
        }
        this.lifecycleScope.launch {
            viewModel.cameraPreviewLocked = true
            val result = stitcher.stitch(
                inputPath,
                panoramaOutputPath,
                CVStitcher.SequentialRotationDescriptor(sequence, HORIZONTAL_ANGLE_STEP_SIZE.toDouble() * PI / 180, is360),
                updateProgress
            )
            viewModel.cameraPreviewLocked = false
            onFinished()
            viewModel.popupMessage.value = if (result.code == CVStitcher.Result.Code.OK)
                PopupMessage(resources.getString(R.string.stitch_succeeded, result.imgPath),
                    Pair(resources.getString(R.string.open)) {
                        try {
                            val newIntent = Intent(Intent.ACTION_VIEW)
                            newIntent.data = FileProvider.getUriForFile(this@MainActivity, resources.getString(R.string.file_provider_authority), File(result.imgPath!!))
                            Log.d("", newIntent.data.toString())
                            newIntent.flags = Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                            startActivity(newIntent)
                        } catch (e: Exception) {

                            viewModel.popupMessage.value = PopupMessage(resources.getString(R.string.failed_to_open))
                        }

                    })
            else
                PopupMessage(resources.getString(R.string.stitch_failed, result.code))

        }


    }

    private fun preparePreview(onPreviewReady: suspend (Boolean) -> Unit) {
        if (!this::cameraPreviewView.isInitialized)
            return

        viewModel.updateCameraPreviewSurfaceProvider(cameraPreviewView.surfaceProvider)

        this.lifecycleScope.launch {
            try {
                viewModel.bindCamera(this@MainActivity)

            }
            catch (ex: TimeoutException) {
                viewModel.popupMessage.value = PopupMessage(resources.getString(R.string.camera_binding_timedout))
                return@launch
            }
            catch (ex: CameraUnavailableException) {
                viewModel.popupMessage.value = PopupMessage(resources.getString(R.string.camera_unavailable))
                return@launch
            }

            val atStart = (viewModel.angleStepIndex.value == 0)
            onPreviewReady(atStart)
            if (atStart)
                viewModel.popupMessage.value = PopupMessage(resources.getString(R.string.aim_at))
        }
    }

    private fun preparePreview(context: Context, onPreviewReady: suspend (Boolean) -> Unit): PreviewView {
        return PreviewView(context).also {
            cameraPreviewView = it
            preparePreview(onPreviewReady)

        }
    }

    @ExperimentalCamera2Interop
    suspend fun lockExposureAndWhiteBalance(on: Boolean) {
        val camera2CameraControl : Camera2CameraControl = Camera2CameraControl.from(viewModel.camera!!.cameraControl)

        val captureRequestOptionsBuilder = CaptureRequestOptions.Builder()
            .setCaptureRequestOption(CaptureRequest.CONTROL_AE_LOCK, on)
            .setCaptureRequestOption(CaptureRequest.CONTROL_AWB_LOCK, on)


        val future = camera2CameraControl.addCaptureRequestOptions(captureRequestOptionsBuilder.build())

        suspendCoroutine { continuation ->
            this.lifecycleScope.launch {
                try {

                    withContext(Dispatchers.IO) {
                        future.get(3000, TimeUnit.MILLISECONDS)
                        Log.d(resources.getString(R.string.app_name), "Camera Control has been applied")
                    }
                }
                catch (e: TimeoutException) {
                    viewModel.popupMessage.value =
                        PopupMessage(resources.getString(R.string.exposure_wb_locking_timedout))
                }

                continuation.resume(Unit)
            }
        }

    }



    private suspend fun preparePictureTaking(): Boolean {

        val meteringPointFactory = cameraPreviewView.meteringPointFactory

        val meteringPoint = meteringPointFactory.createPoint(cameraPreviewView.measuredWidth.toFloat()*0.5f, cameraPreviewView.measuredHeight.toFloat()*0.5f)


        val focusMeteringBuilder = FocusMeteringAction.Builder(meteringPoint, FocusMeteringAction.FLAG_AF)
        /*
        if (stageIndex == 0) {
            FocusMeteringAction.Builder(meteringPoint)
                .addPoint(meteringPointFactory.createPoint(cameraPreviewView.measuredWidth.toFloat()*0.25f, cameraPreviewView.measuredHeight.toFloat()*0.25f), FocusMeteringAction.FLAG_AE or FocusMeteringAction.FLAG_AWB)
                .addPoint(meteringPointFactory.createPoint(cameraPreviewView.measuredWidth.toFloat()*0.75f, cameraPreviewView.measuredHeight.toFloat()*0.25f), FocusMeteringAction.FLAG_AE or FocusMeteringAction.FLAG_AWB)
                .addPoint(meteringPointFactory.createPoint(cameraPreviewView.measuredWidth.toFloat()*0.25f, cameraPreviewView.measuredHeight.toFloat()*0.75f), FocusMeteringAction.FLAG_AE or FocusMeteringAction.FLAG_AWB)
                .addPoint(meteringPointFactory.createPoint(cameraPreviewView.measuredWidth.toFloat()*0.75f, cameraPreviewView.measuredHeight.toFloat()*0.75f), FocusMeteringAction.FLAG_AE or FocusMeteringAction.FLAG_AWB)

        }
        else
            FocusMeteringAction.Builder(meteringPoint, FocusMeteringAction.FLAG_AF)

        */


        val focusMeteringAction = focusMeteringBuilder.disableAutoCancel().build()

        val fmResult = viewModel.camera!!.cameraControl.startFocusAndMetering(focusMeteringAction)
        return suspendCoroutine { continuation ->
            this.lifecycleScope.launch {

                var successful = false
                try {
                    withContext(Dispatchers.IO) {
                        successful = fmResult.get(5000, TimeUnit.MILLISECONDS).isFocusSuccessful
                        Log.d(
                            getString(R.string.app_name),
                            "startFocus callback: successful = $successful"
                        )
                    }
                }
                catch (ex: TimeoutException) {
                    viewModel.popupMessage.value = PopupMessage(resources.getString(R.string.auto_focusing_timedout))
                    successful = true
                }
                catch (ex: ExecutionException) {
                    Log.d(getString(R.string.app_name), "startFocus threw ExecutionException: ${ex.message}")
                }
                catch (ex: CameraControl.OperationCanceledException) {
                    Log.d(getString(R.string.app_name), "startFocus threw OperationCanceledException: ${ex.message}")
                }
                continuation.resume(successful)

            }
        }
    }


    private fun generateTempImageName(stepIndex: Int): String {
        return "${stepIndex}.jpg"
    }



    private inline fun takePicture(crossinline callback: suspend (Boolean, Boolean) -> Unit)
    {

        Log.d(getString(R.string.app_name), "TakePicture() starts")

        val outputFileOptions = ImageCapture.OutputFileOptions.Builder(
            File("${tempPath}/"+generateTempImageName(viewModel.angleStepIndex.value))
        ).build()



        viewModel.imageCapture.takePicture(outputFileOptions, viewModel.backgroundExecutor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(error: ImageCaptureException)
                {
                    this@MainActivity.lifecycleScope.launch{

                        Log.d(getString(R.string.app_name), "Failed to capture an image: ${error.message}")
                        callback(false, true)
                        viewModel.popupMessage.value = PopupMessage(resources.getString(
                            R.string.taking_picture_error
                        ))

                    }
                }
                @ExperimentalCamera2Interop
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {

                    this@MainActivity.lifecycleScope.launch{
                        Log.d(getString(R.string.app_name), "A new image has been captured")


                        if (viewModel.angleStepIndex.value == 0)
                            lockExposureAndWhiteBalance(true)

                        viewModel.updateLookatAngle(deviceOrientationMonitor)

                        callback(true, viewModel.angleStepIndex.value != NUM_OF_ANGLE_STEPS)


                    }
                }
            })




    }
}

