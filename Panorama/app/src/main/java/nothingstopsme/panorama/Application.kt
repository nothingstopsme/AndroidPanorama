package nothingstopsme.panorama


import androidx.camera.camera2.Camera2Config
import androidx.camera.core.CameraXConfig
import 	android.app.Application as AndroidApplication



class Application : AndroidApplication(), CameraXConfig.Provider  {




    override fun getCameraXConfig(): CameraXConfig {
        return CameraXConfig.Builder.fromConfig(Camera2Config.defaultConfig())
            .build()
    }


}