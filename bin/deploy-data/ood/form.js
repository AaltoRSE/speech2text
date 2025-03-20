/**
 *  Decode the audio_path to replace '%20' with space
 */

// File path in OOD has a prefix
const OOD_PREFIX_PATH = "/pun/sys/dashboard/files/fs/";

function decode_audio_path() {
    // Do nothing if a folder is selected (no files are selected)
    if ($("#batch_connect_session_context_audio_path_path_selector_table tr.selected").length === 0) {
        return; 
    }
    
    let selectedFiles = [];
    $("#batch_connect_session_context_audio_path_path_selector_table tr.selected").each(function() {
        // Extract the file path from the data attribute
        let filePath = $(this).data("api-url");
        if (filePath) {
            let absolutePath = filePath.replace(OOD_PREFIX_PATH, ''); // Remove the prefix from the path
            selectedFiles.push(decodeURIComponent(absolutePath));
        }
    });

    // Join paths with a separator, e.g., comma
    $("#batch_connect_session_context_audio_path").val(selectedFiles.join(', '));

}


function toggle_data_warning(isChecked) {
    // Select the label associated with the checkbox
    const label = $("label[for='batch_connect_session_context_send_attachments']");
    const warningMessage = `
            <div id="confidential-warning" style="color: blue; margin-top: 5px;">
                We recommed this only if your audio files do not include any confidential data.
            </div>
    `;
    if (isChecked) {
        // Add a border box around the form group
        label.closest('.form-group').css({
            "border": "1px solid blue",
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


function toggle_visibilty_of_form_group(form_id, show) {
    let form_element = $(form_id);
    let parent = form_element.parent();
    console.log("Show value:", show);
    if(show == true) {
      parent.show();
    } else {
      parent.hide();
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

    let advance_settings = $("#batch_connect_session_context_advance_options");
    advance_settings.change(function() {
        toggle_visibilty_of_form_group(
            "#batch_connect_session_context_model_selector", 
            advance_settings.is(':checked'))
    });
}


/**
 *  Install event handlers
 */
$(document).ready(function () {
    add_event_handlers();

    // Hide the advance settings at the beggining
    toggle_visibilty_of_form_group("#batch_connect_session_context_model_selector", 'false')
});