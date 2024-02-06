package nothingstopsme.panorama

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import org.bytedeco.opencv.global.opencv_core

class CVAsyncWorker(estimatedNumber: Int, private val scope: CoroutineScope) {
    private val jobs = ArrayList<Deferred<Unit>>(estimatedNumber)

    /*
     * Note: because the internal flag for "UseOpenCL" is a thread-local variable,
     * to make sure every thread observes the same value,
     * the current flag is cached before operations which might trigger thread-switching,
     * e.g. starting an asynchronous coroutine or waiting for a suspension point in a coroutine to finish,
     * and the cached flag is reapplied when the execution enters the context which might be run
     * by a different thread
     */




    companion object {

        suspend inline fun <T> waitFor(crossinline waitable: suspend () -> T ): T {
            val useOpenCL = opencv_core.useOpenCL()
            val result = waitable()
            opencv_core.setUseOpenCL(useOpenCL)
            return result
        }
    }

    fun add(job: suspend CoroutineScope.() -> Unit){
        val useOpenCL = opencv_core.useOpenCL()
        jobs.add(scope.async {
            opencv_core.setUseOpenCL(useOpenCL)
            job()
        })
    }

    suspend fun awaitAll() {
        val useOpenCL = opencv_core.useOpenCL()
        jobs.awaitAll()
        jobs.clear()
        opencv_core.setUseOpenCL(useOpenCL)
    }
}