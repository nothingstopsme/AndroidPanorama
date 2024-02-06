package nothingstopsme.panorama

import android.content.Context
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.CameraUnavailableException
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.geometry.Offset
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

data class PopupMessage(private val message: String, val action: Pair<String, suspend () -> Unit>?=null) {
    var isRead = false
        private set

    fun read(): String {
        isRead = true
        return message
    }
}

class MainViewModel : ViewModel() {

    private class CameraLifecycleOwner: LifecycleOwner {
        private val lifecycleImpl = LifecycleRegistry(this)

        override val lifecycle: Lifecycle
            get() {
                return lifecycleImpl
            }

        init {
            lifecycleImpl.currentState = Lifecycle.State.STARTED
        }

        fun finish() {
            lifecycleImpl.currentState = Lifecycle.State.DESTROYED
        }
    }


    private var cameraLifecycleOwner = CameraLifecycleOwner()



    val backgroundExecutor: ExecutorService = Executors.newCachedThreadPool()

    private val _targetOffset = mutableStateOf(Offset(0f, 0f))
    val targetOffset: State<Offset>
        get() = _targetOffset


    val readyFlag = mutableStateOf(false)
    val popupMessage = mutableStateOf(PopupMessage(""))



    private val _angleStepIndex = mutableIntStateOf(0)
    val angleStepIndex: State<Int>
        get() = _angleStepIndex


    private var lookAt: DeviceOrientationMonitor.LookAt? = null

    var camera: Camera? = null
        private set


    private val cameraPreview = Preview.Builder().build()

    private var cameraPreviewSurfaceProvider : Preview.SurfaceProvider? = null

    var cameraPreviewLocked = false
        set(value) {
            if (value)
                cameraPreview.setSurfaceProvider(null)
            else
                cameraPreview.setSurfaceProvider(cameraPreviewSurfaceProvider)

            field = value
        }

    val imageCapture = ImageCapture.Builder()
        .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
        .setIoExecutor(backgroundExecutor)
        .setJpegQuality(100)
        .build()

    private val cameraSelector = CameraSelector.Builder()
        .requireLensFacing(CameraSelector.LENS_FACING_BACK).build()

    fun updateCameraPreviewSurfaceProvider(value: Preview.SurfaceProvider?) {
        cameraPreviewSurfaceProvider = value
        if (!cameraPreviewLocked)
            cameraPreview.setSurfaceProvider(cameraPreviewSurfaceProvider)
    }


    suspend fun bindCamera(context: Context) {
        if (camera == null) {
            val cameraProviderFuture = ProcessCameraProvider.getInstance(context)


            val cameraProvider = withContext(Dispatchers.IO) {
                cameraProviderFuture.get(3000, TimeUnit.MILLISECONDS)
            }

            camera = cameraProvider.bindToLifecycle(
                cameraLifecycleOwner,
                cameraSelector,
                cameraPreview,
                imageCapture
            )



        }

        if (camera == null)
            throw CameraUnavailableException(CameraUnavailableException.CAMERA_UNKNOWN_ERROR)

    }

    override fun onCleared() {
        cameraLifecycleOwner.finish()
        super.onCleared()
    }

    suspend fun updateLookatAngle(deviceOrientationMonitor: DeviceOrientationMonitor) {

        val (latestLookAt, deviceToWorldRotation) = deviceOrientationMonitor.getLatestLookAt()
        if (_angleStepIndex.intValue == 0) {
            lookAt = latestLookAt
        }

        if(_angleStepIndex.intValue < MainActivity.NUM_OF_ANGLE_STEPS -1)
        {
            lookAt!!.rotateAroundZ(-MainActivity.HORIZONTAL_ANGLE_STEP_SIZE)
            _angleStepIndex.intValue += 1
        }
        else {

            _angleStepIndex.intValue = MainActivity.NUM_OF_ANGLE_STEPS
        }
    }



    fun updateTargetOffset(deviceOrientationMonitor: DeviceOrientationMonitor) {
        lookAt?.let {
            _targetOffset.value = deviceOrientationMonitor.computeOffsetsInDevice(it)
        }
    }



    fun reset() {
        lookAt = null

        _targetOffset.value = Offset.Zero
        _angleStepIndex.intValue = 0


        camera?.cameraControl?.cancelFocusAndMetering()
    }


    fun releaseCamera() {
        cameraLifecycleOwner.finish()
        cameraLifecycleOwner = CameraLifecycleOwner()
        camera = null
    }

}