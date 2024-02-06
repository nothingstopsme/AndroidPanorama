package nothingstopsme.panorama

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.opengl.Matrix
import androidx.compose.ui.geometry.Offset
import kotlin.coroutines.Continuation
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine
import kotlin.math.max
import kotlin.math.min

class DeviceOrientationMonitor : SensorEventListener
{

    class LookAt private constructor()
    {
        val lookAt = FloatArray(4)
        val side = FloatArray(4)


        constructor(v:FloatArray) : this() {
            v.copyInto(lookAt)

            if(lookAt[0] == 0f && lookAt[1] == 0f)
                side[0] = 1f
            else {
                side[0] = lookAt[1]
                side[1] = -lookAt[0]
            }
            side[3] = 1f


        }



        fun copy() : LookAt {
            val copy = LookAt()
            lookAt.copyInto(copy.lookAt)
            side.copyInto(copy.side)

            return copy
        }


        fun rotateAroundZ(degree: Float) {
            val transformation = FloatArray(16)

            Matrix.setRotateM(transformation, 0, degree, 0f, 0f, 1f)
            Matrix.multiplyMV(lookAt, 0, transformation, 0, lookAt, 0)
            Matrix.multiplyMV(side, 0, transformation, 0, side, 0)

        }

        fun rotateAroundSide(degree: Float) {
            val transformation = FloatArray(16)

            Matrix.setRotateM(transformation, 0, degree, side[0], side[1], side[2])
            Matrix.multiplyMV(lookAt, 0, transformation, 0, lookAt, 0)

        }

    }

    private var updateCallback: (DeviceOrientationMonitor.() -> Unit)? = null

    private val worldToDeviceRotation = FloatArray(16)



    private val inDeviceLookAt = floatArrayOf(0f, 0f, -1f, 1f)

    private val waitingList = mutableListOf<Continuation<Pair<LookAt, FloatArray>>>()



    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onSensorChanged(event: SensorEvent?) {
        if(event != null) {

            /*
             * The matrix retrieved from getRotationMatrixFromVector() is actually
             * the device-to-world rotation matrix;
             * however, because the interpretation of android.opengl.Matrix is adopted, which views a array of entries
             * in column-major order, the matrix in effect yields a world-to-device mapping
             */
            SensorManager.getRotationMatrixFromVector(worldToDeviceRotation, event.values)




            if(waitingList.isNotEmpty()) {
                val inWorldLookAt = FloatArray(4)
                val deviceToWorldRotation = FloatArray(16)
                Matrix.transposeM(deviceToWorldRotation, 0, worldToDeviceRotation, 0)
                Matrix.multiplyMV(inWorldLookAt, 0, deviceToWorldRotation, 0, inDeviceLookAt, 0)

                val lookAt = LookAt(inWorldLookAt)

                for (index in 1 until waitingList.size) {
                    waitingList[index].resume(Pair(lookAt.copy(), deviceToWorldRotation.clone()))
                }
                waitingList[0].resume(Pair(lookAt, deviceToWorldRotation))
                waitingList.clear()
            }


            updateCallback?.invoke(this)
        }
    }

    suspend fun getLatestLookAt(): Pair<LookAt, FloatArray> {
        return suspendCoroutine { continuation -> waitingList.add(continuation) }
    }

    fun computeOffsetsInDevice(reference: LookAt): Offset
    {
        val inDevice = FloatArray(4)
        Matrix.multiplyMV(inDevice, 0, worldToDeviceRotation, 0, reference.lookAt, 0)


        // A positive y value means moving upward, so when described as y-offset on the screen its value should be negated
        return Offset(max(min(inDevice[0], 1f), -1f), max(min(-inDevice[1], 1f), -1f))
    }

    fun onUpdate(updateCallback: (DeviceOrientationMonitor.() -> Unit)?){
        this.updateCallback = updateCallback
    }

}