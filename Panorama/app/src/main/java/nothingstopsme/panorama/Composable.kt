package nothingstopsme.panorama

import android.annotation.SuppressLint
import android.content.Context
import android.os.Parcel
import android.os.Parcelable
import android.view.View
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColor
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateIntOffsetAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box

import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset

import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack

import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.SnackbarResult
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField

import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisallowComposableCalls
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.MutableFloatState
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.State
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalInspectionMode
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.KeyboardType
import androidx.constraintlayout.compose.ConstraintLayout
import androidx.constraintlayout.compose.Dimension
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import nothingstopsme.panorama.ui.theme.PanoramaTheme
import nothingstopsme.panorama.ui.theme.extraScheme
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt



@Composable
fun Target(modifier: Modifier, targetOffset: State<Offset>, offsetScaleInPx: Float)
{
    val targetIntOffsetAnimated = animateIntOffsetAsState(
        IntOffset(
            (targetOffset.value.x * offsetScaleInPx).roundToInt(),
            (targetOffset.value.y * offsetScaleInPx).roundToInt()
        ),
        animationSpec = spring(stiffness = Spring.StiffnessLow)
    )

    val targetTransition = rememberInfiniteTransition()
    val targetColor = targetTransition.animateColor(
        initialValue = Color.Black,
        targetValue = Color.White,
        animationSpec = infiniteRepeatable(
            animation = tween(500, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        )
    )

    Image(
        modifier = modifier.offset {
            targetIntOffsetAnimated.value
        },
        painter = painterResource(R.drawable.target),
        colorFilter = ColorFilter.tint(targetColor.value),
        contentDescription = "target",
    )
}

@Composable
inline fun CrossHairs(modifier: Modifier, enabled: State<Boolean>, onTarget: State<Boolean>,
                      crossinline startLocking: suspend () -> Boolean,
                      crossinline onUnlocked: () -> Unit,
                      crossinline onLocked: () -> Unit)
{
    val lockingSweepAngle = remember { Animatable(0f) }
    LaunchedEffect(enabled.value, onTarget.value) {

        if (enabled.value && onTarget.value) {

            try {

                val lockingDeferred = async {
                    return@async startLocking()
                }

                while (true) {
                    lockingSweepAngle.snapTo(0f)
                    lockingSweepAngle.animateTo(
                        360f,
                        animationSpec = tween(
                            durationMillis = 1500,
                            easing = LinearEasing
                        )
                    )

                    if (!lockingDeferred.isActive) {
                        lockingDeferred.await()
                        break
                    }
                }


                onLocked()

            } catch (ex: CancellationException) {
                onUnlocked()
            }

        }
        else {
            val isRunning = lockingSweepAngle.isRunning
            lockingSweepAngle.snapTo(0f)
            if (!isRunning)
                onUnlocked()
        }

    }

    val circleStrokeWidth = 2f * LocalDensity.current.density
    Image(
        modifier = modifier.drawBehind {
            drawArc(
                Color.Green, -90f, lockingSweepAngle.value, false,
                topLeft = Offset(circleStrokeWidth, circleStrokeWidth),
                size = Size(this.size.width-circleStrokeWidth*2f, this.size.height-circleStrokeWidth*2f),
                style = Stroke(width=circleStrokeWidth, cap= StrokeCap.Square)
            )
        },
        painter = painterResource(R.drawable.crosshairs),
        colorFilter = if (onTarget.value) null else ColorFilter.tint(Color(0xFFB60000)),
        contentDescription = "crosshairs",
    )
}

@Composable
fun Option(text:String, backgroundColor: Color, action: () -> Unit) {

    TextButton(
        modifier = Modifier
            .fillMaxWidth()
            .height(50.dp),
        colors = ButtonDefaults.textButtonColors(
            containerColor = backgroundColor,
            contentColor = MaterialTheme.colorScheme.primary
        ),
        shape = RectangleShape,
        onClick = action
    ){
        ConstraintLayout(modifier = Modifier.fillMaxWidth()) {
            val textRef = createRef()
            Text(text, modifier = Modifier.constrainAs(textRef){
                start.linkTo(parent.start, margin = 5.dp)
            })
        }
    }

}

data class MainScreenStates(
    val progressBarVisible: MutableState<Boolean>,
    val lockingEnabled: MutableState<Boolean>,
    val curtainVisible: MutableState<Boolean>,
    val currentProgress: MutableFloatState,
    val menuVisible: MutableState<Boolean>
) : Parcelable {

    private constructor(input: Parcel) : this(
        mutableStateOf(input.readByte() > 0),
        mutableStateOf(input.readByte() > 0),
        mutableStateOf(input.readByte() > 0),
        mutableFloatStateOf(input.readFloat()),
        mutableStateOf(input.readByte() > 0)
    )

    override fun describeContents(): Int {
        return 0
    }

    override fun writeToParcel(dest: Parcel, flags: Int) {
        dest.writeByte(if (progressBarVisible.value) 1.toByte() else 0.toByte())
        dest.writeByte(if (lockingEnabled.value) 1.toByte() else 0.toByte())
        dest.writeByte(if (curtainVisible.value) 1.toByte() else 0.toByte())
        dest.writeFloat(currentProgress.floatValue)
        dest.writeByte(if (menuVisible.value) 1.toByte() else 0.toByte())
    }

    companion object CREATOR: Parcelable.Creator<MainScreenStates?> {
        override fun createFromParcel(input: Parcel): MainScreenStates {
            return MainScreenStates(input)
        }

        override fun newArray(size: Int): Array<MainScreenStates?> {
            return arrayOfNulls(size)
        }
    }
}

@SuppressLint("UnusedMaterial3ScaffoldPaddingParameter")
@OptIn(ExperimentalMaterial3Api::class)
@Composable
inline fun <PREVIEW: View> MainScreenStates.MainScreen(
    navController: NavHostController,
    popupMessage: State<PopupMessage>,
    readyFlag: State<Boolean>,
    targetOffset: State<Offset>,
    angleStepIndex: State<Int>,
    crossinline preparePreview: (Context, suspend (Boolean) -> Unit) -> PREVIEW,
    crossinline preparePictureTaking: suspend () -> Boolean,
    crossinline takePicture: (suspend (Boolean, Boolean) -> Unit) -> Unit,
    crossinline stitch: @DisallowComposableCalls (updateProgress: suspend (Float) -> Unit, onFinished: suspend () -> Unit) -> Unit,
    crossinline reset: @DisallowComposableCalls suspend() -> Unit

) {
    val isPreviewing = LocalInspectionMode.current
    val coroutineScope = rememberCoroutineScope()


    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        val snackbarHostState = remember { SnackbarHostState() }
        val takingPicture = remember {  mutableStateOf(false) }


        val restart = remember {

            suspend {
                lockingEnabled.value = false
                reset()


                //takingPicture = false

                progressBarVisible.value = false
                curtainVisible.value = false
            }
        }

        val stitchNow = remember {
            {
                //val lockingEnabledCache = lockingEnabled.value
                lockingEnabled.value = false
                curtainVisible.value = true

                currentProgress.floatValue = 0.0f
                progressBarVisible.value = true

                stitch( { progress ->
                    withContext(coroutineScope.coroutineContext) {
                        currentProgress.floatValue = progress
                    }
                }, {

                    withContext(coroutineScope.coroutineContext) {
                        restart()
                        //progressBarVisible.value = false
                        //curtainVisible.value = false
                        //lockingEnabled.value = lockingEnabledCache

                    }

                })
            }
        }


        Scaffold(
            modifier = Modifier.fillMaxSize(),
            snackbarHost = { SnackbarHost(snackbarHostState) })  {

            LaunchedEffect(popupMessage.value) {
                if(!popupMessage.value.isRead) {
                    val message = popupMessage.value.read()
                    if (message.isNotEmpty()) {
                        if (popupMessage.value.action == null)
                            snackbarHostState.showSnackbar(message)
                        else {
                            val result = snackbarHostState.showSnackbar(
                                message,
                                popupMessage.value.action!!.first,
                                true,
                                SnackbarDuration.Indefinite
                            )

                            if (result == SnackbarResult.ActionPerformed)
                                popupMessage.value.action!!.second()

                        }
                    }
                }
            }



            BoxWithConstraints(
                modifier = Modifier.fillMaxSize()
            ) {

                AndroidView(

                    modifier = Modifier
                        .matchParentSize()
                        .pointerInput(Unit) {
                            detectTapGestures(
                                onPress = {

                                    menuVisible.value = false
                                    if (angleStepIndex.value == 0) {
                                        lockingEnabled.value = true
                                        tryAwaitRelease()
                                        if (angleStepIndex.value == 0 && !takingPicture.value)
                                            lockingEnabled.value = false
                                    }


                                },
                                onLongPress = {
                                    if (angleStepIndex.value > 0 && !takingPicture.value)
                                        menuVisible.value = true
                                }
                            )


                        },
                    factory = { context ->

                        preparePreview(context) { atStart ->
                            withContext(coroutineScope.coroutineContext) {
                                //resetPanoramaStage()
                                if (atStart) {
                                    restart()
                                }

                            }
                        }

                    }
                )


                if (!progressBarVisible.value) {

                    val minBoxSideHalfLengthInPx = min(
                        this@BoxWithConstraints.maxWidth.value,
                        this@BoxWithConstraints.maxHeight.value
                    ) * 0.5f * LocalDensity.current.density




                    Target(Modifier.align(Alignment.Center), targetOffset, minBoxSideHalfLengthInPx)


                    val onTarget = remember {
                        derivedStateOf {
                            return@derivedStateOf (targetOffset.value.x.pow(2f) + targetOffset.value.y.pow(
                                2f
                            ) <= 5e-3)
                        }
                    }


                    CrossHairs(Modifier.align(Alignment.Center), lockingEnabled, onTarget,
                        startLocking = {
                            preparePictureTaking()
                        },
                        onUnlocked = {
                        },
                        onLocked = {

                            takingPicture.value = true
                            takePicture { ok, need_more ->

                                withContext(coroutineScope.coroutineContext) {
                                    takingPicture.value = false
                                    if (angleStepIndex.value == 0)
                                        lockingEnabled.value = false

                                    if (!need_more)
                                        stitchNow()
                                    else if (ok) {
                                        curtainVisible.value = true
                                        // Creating a blinking effect by removing the curtain with a small delay
                                        if (!isPreviewing)
                                            delay(200)
                                        curtainVisible.value = false
                                    }


                                }
                            }

                        })



                    AnimatedVisibility(
                        visible = menuVisible.value,
                        modifier = Modifier.align(Alignment.Center)
                    ) {
                        Column(
                            modifier = Modifier
                                .width((this@BoxWithConstraints.maxWidth.value * 0.7).dp)
                        ) {

                            Option("Stitch now", MaterialTheme.colorScheme.background) {
                                menuVisible.value = false
                                stitchNow()
                            }
                            Option("Restart", MaterialTheme.colorScheme.extraScheme.background2) {

                                menuVisible.value = false
                                curtainVisible.value = true
                                coroutineScope.launch {
                                    //resetPanoramaStage()
                                    restart()
                                }


                            }


                        }
                    }

                    ConstraintLayout(modifier = Modifier.fillMaxSize()) {
                        val iconRef = createRef()
                        IconButton(
                            onClick = {
                                navController.navigate("settings")
                            },
                            modifier = Modifier.constrainAs(iconRef) {

                                end.linkTo(parent.end, margin = 5.dp)
                                top.linkTo(parent.top, margin = 5.dp)

                            }) {
                            Image(
                                Icons.Filled.Settings,
                                null,
                                modifier = Modifier.size(60.dp),
                                colorFilter = ColorFilter.tint(MaterialTheme.colorScheme.extraScheme.settings)
                            )
                        }
                    }

                }
            }


            Box(
                modifier = Modifier.fillMaxSize()
            ) {
                if (!readyFlag.value || curtainVisible.value) {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(MaterialTheme.colorScheme.extraScheme.translucentBackground)
                            .pointerInput(Unit) {
                                detectTapGestures()
                            }
                    )
                    {}
                }
                if (progressBarVisible.value) {
                    LinearProgressIndicator(
                        progress = currentProgress.floatValue,
                        modifier = Modifier
                            .align(Alignment.Center)
                            .fillMaxWidth(0.9f)
                            .height(4.dp),
                        color = MaterialTheme.colorScheme.extraScheme.progressIndicator,
                        trackColor = MaterialTheme.colorScheme.background,
                    )
                }
            }
        }

    }

}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(navController: NavHostController, preferences:  UserPreferencesInterface){

    val coroutineScope = rememberCoroutineScope()

    val minMatchConfidence = remember {
        mutableStateOf("")
    }

    /*
     * Initialising the content of minMatchConfidence in a coroutine,
     * so that the value from preferences provided through flows can be read out
     */
    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            minMatchConfidence.value = preferences.minMatchConfidence.first().toString()
        }

    }


    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {

        ConstraintLayout(modifier = Modifier.fillMaxSize()) {
            val columnRef = createRef()


            Column(modifier = Modifier.constrainAs(columnRef) {
                start.linkTo(parent.start, margin = 5.dp)
                end.linkTo(parent.end, margin = 5.dp)
                top.linkTo(parent.top, margin = 5.dp)
                bottom.linkTo(parent.bottom)
                height = Dimension.fillToConstraints
                width = Dimension.fillToConstraints

            }) {

                IconButton(
                    onClick = {
                        navController.navigate("main")
                    }) {
                    Image(
                        Icons.Filled.ArrowBack,
                        null,
                        modifier = Modifier.size(60.dp),
                        colorFilter = ColorFilter.tint(MaterialTheme.colorScheme.extraScheme.back)
                    )
                }



                val floatPattern = remember { Regex("^\\d+(\\.\\d*)?|\$") }

                TextField(
                    minMatchConfidence.value,
                    onValueChange = {
                        /*
                         * Regular Expression is used to constrain user inputs.
                         * since a slightly more freedom is granted to users during editing,
                         * toFloatOrNull() might return null even when an input passes the check of floatPattern,
                         * and only when considered valid by both checks can an input be written back into the preferences
                         */
                        if (floatPattern.matches(it)) {
                            minMatchConfidence.value = it
                            it.toFloatOrNull()?.also {
                                coroutineScope.launch(Dispatchers.IO) {

                                    preferences.setMinMatchConfidence(it)
                                }
                            }

                        }


                    }, supportingText = {
                        Text(text = stringResource(R.string.min_match_confidence_support))
                    },
                    label = {
                        Text(text = stringResource(R.string.min_match_confidence))
                    },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(modifier = Modifier
                    .fillMaxWidth()
                    .height(5.dp))

                ConstraintLayout(modifier = Modifier.fillMaxWidth()) {
                    val labelRef = createRef()
                    val switchRef = createRef()

                    Text(stringResource(R.string.open_cl_enabled), Modifier.constrainAs(labelRef){
                        centerVerticallyTo(parent)
                    })

                    Switch(
                        checked = preferences.openCLEnabled.collectAsState(initial = false, Dispatchers.IO).value,
                        onCheckedChange = {
                            coroutineScope.launch(Dispatchers.IO) {
                                preferences.setOpenCLEnabled(it)
                            }

                        },
                        modifier = Modifier.constrainAs(switchRef) {
                            end.linkTo(parent.end)
                        }
                    )
                }
            }

        }

    }

}


@Composable
inline fun Navigation(crossinline main: @Composable (MainScreenStates.(navController: NavHostController)->Unit),
                      crossinline settings: @Composable ((navController: NavHostController)->Unit)) {

    val navController = rememberNavController()
    val mainScreenStates = rememberSaveable {
        MainScreenStates(
            mutableStateOf(false),
            mutableStateOf(false),
            mutableStateOf(true),
            mutableFloatStateOf(0.0f),
            mutableStateOf(false)
        )
    }
    NavHost(navController = navController, startDestination = "main") {
        composable("main") {
            mainScreenStates.main(navController)
        }
        composable("settings") {
            settings(navController)
        }
    }

}


@Preview(showBackground = true,
    backgroundColor = 0xFF00FF00,
    //uiMode = Configuration.UI_MODE_NIGHT_YES,

)
@Composable
fun FullPreview() {

    val popupMessage = remember { mutableStateOf(PopupMessage("")) }
    val readyFlag = remember { mutableStateOf(true) }

    val targetOffset = remember{ mutableStateOf(Offset(0f, 0f)) }
    val angleStepIndex = remember { mutableIntStateOf(0) }
    val coroutineScope = rememberCoroutineScope()

    val fakePreferences = remember {
        object: UserPreferencesInterface {

            private val minMatchConfidenceFlow = MutableStateFlow(0.75f)
            private val openCLEnabledFlow = MutableStateFlow(true)

            override val minMatchConfidence: Flow<Float>
                get() = minMatchConfidenceFlow

            override suspend fun setMinMatchConfidence(value: Float) {
                minMatchConfidenceFlow.value = value
            }

            override val openCLEnabled: Flow<Boolean>
                get() = openCLEnabledFlow

            override suspend fun setOpenCLEnabled(value: Boolean) {
                openCLEnabledFlow.value = value
            }
        }
    }
    PanoramaTheme(darkTheme = true) {

        Navigation(
            main = {
                MainScreen(
                    it,
                    popupMessage,
                    readyFlag,
                    targetOffset,
                    angleStepIndex,
                    preparePreview = { context, callback ->
                        coroutineScope.launch {
                            callback(true)
                            popupMessage.value = PopupMessage("Hello")
                        }
                        View(context)
                    },
                    preparePictureTaking = {
                        true
                    },
                    takePicture = { callback ->
                        coroutineScope.launch {
                            angleStepIndex.intValue += 1
                            targetOffset.value = Offset(1.0f, 0f)
                            callback(true, true)
                        }
                    },
                    stitch = { updateProgress, reportResult ->
                    },
                    reset = {
                        targetOffset.value = Offset(0f, 0f)
                        angleStepIndex.intValue = 0
                    }
                )
            },
            settings = {
                SettingsScreen(it, fakePreferences)
            }
        )
    }

}