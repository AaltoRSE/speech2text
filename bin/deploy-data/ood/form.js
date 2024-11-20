/**
 *  Decode the audio_path to replace '%20' with space
 */
function decode_audio_path() {
    let audio_path = $("#batch_connect_session_context_audio_path");

    audio_path.val(decodeURIComponent(audio_path.val()))
}


function toggle_data_warning(isChecked) {
    // Select the label associated with the checkbox
    const label = $("label[for='batch_connect_session_context_send_attachments']");
    const warningMessage = `
            <div id="confidential-warning" style="color: red; margin-top: 5px;">
                Warning: We don't recommend sending confidential data via email.
            </div>
    `;
    if (isChecked) {
        // Add a border box around the form group
        label.closest('.form-group').css({
            "border": "1px solid red",
            "padding": "2px"
        });
        $("#batch_connect_session_context_send_attachments")
            .closest('.form-group')
            .append(warningMessage);
    } else {
        
        $("#confidential-warning").remove();
        // Reset the label color and border to its default
        label.css("color", "");
        label.closest('.form-group').css({
            "border": "",
            "padding": ""
        });
    }
}


function validate_AudioPath(event) {
    let audio_path = $("#batch_connect_session_context_audio_path").val();
    if (!audio_path) {
        event.preventDefault();
        alert("The audio path field cannot be empty.");
    }
}


/**
 * Sets the event handler for file selector button.
 * Triggering the handler based on the field change doesn't work
 * as the field doesn't get the focus.
 */
function add_event_handlers() {
    let audio_path_button = $(
        "#batch_connect_session_context_audio_path_path_selector_button"
    );
    audio_path_button.click(decode_audio_path);
    
    let email_checkbox = $(
        "#batch_connect_session_context_send_attachments"
    );
    email_checkbox.change(function() {
        toggle_data_warning(email_checkbox.is(':checked'));
    });

    let submit_button = $("input[type='submit'][name='commit']");
    submit_button.click(validate_AudioPath);
}


/**
 *  Install event handlers
 */
$(document).ready(function () {
    add_event_handlers();
});