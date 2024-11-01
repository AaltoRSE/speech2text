/**
 *  Decode the audio_path to replace '%20' with space
 */
function decode_audio_path() {
    let audio_path = $("#batch_connect_session_context_audio_path");

    audio_path.val(decodeURIComponent(audio_path.val()))
}

/**
 * Sets the event handler for file selector button.
 * Triggering the handler based on the field change doesn't work
 * as the field doesn't get the focus.
 */
function set_audio_path_handler() {
    let audio_path_button = $(
        "#batch_connect_session_context_audio_path_path_selector_button"
    );
    audio_path_button.click(decode_audio_path);
}

/**
 *  Install event handlers
 */
$(document).ready(function () {
    set_audio_path_handler();
});