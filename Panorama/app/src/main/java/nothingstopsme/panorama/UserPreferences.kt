package nothingstopsme.panorama

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "user_preferences")

interface UserPreferencesInterface {
    val minMatchConfidence: Flow<Float>
    suspend fun setMinMatchConfidence(value: Float)

    val openCLEnabled: Flow<Boolean>
    suspend fun setOpenCLEnabled(value: Boolean)
}

class UserPreferences(private val context: Context) : UserPreferencesInterface {

    companion object {
        private val MIN_MATCH_CONFIDENCE_KEY = floatPreferencesKey("min_match_confidence")
        private val OPEN_CL_ENABLED_KEY = booleanPreferencesKey("open_cl_enabled")
    }


    override val minMatchConfidence: Flow<Float>
        get() {
            return context.dataStore.data.map { preferences ->
                (preferences[MIN_MATCH_CONFIDENCE_KEY] ?: 1.0f)
            }
        }

    override suspend fun setMinMatchConfidence(value: Float) {

        context.dataStore.edit { settings ->
            settings[MIN_MATCH_CONFIDENCE_KEY] = value
        }
    }

    override val openCLEnabled: Flow<Boolean>
        get() {
            return context.dataStore.data.map { preferences ->
                (preferences[OPEN_CL_ENABLED_KEY] ?: true)
            }
        }

    override suspend fun setOpenCLEnabled(value: Boolean) {
        context.dataStore.edit { settings ->
            settings[OPEN_CL_ENABLED_KEY] = value
        }
    }
}