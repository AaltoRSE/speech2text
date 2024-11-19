import smtplib
from email.mime.text import MIMEText


AALTO_SMTP_SERVER = 'smtp.aalto.fi'


def sendemail(to: str, 
              file_name: str,
              file_path: str,
              subject: str,
              sender: str,
              job_id: int = None):

    body = f"""
    <html>
        <body>
            <p>Hi,</p>
            <p>Your transcription job for file '{file_name}' is now completed.</p>
            <p>Transcripted files are available inside the <a href="{get_ood_url(file_path)}">results folder</a>.</p>
    """

    if job_id:
        body += f"<p>Your job ID (for diagnosis) was {job_id}.</p>"
    
    body += """
            <p>This is an auto-generated email. Please do not reply to this. If you want to contact our team, please visit our <a href="https://scicomp.aalto.fi/help/garage/">daily garage</a>, every day at 13:00 EET.</p>
        </body>
    </html>
    """

    msg = MIMEText(body, 'html')

    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = to
    
    smtp = smtplib.SMTP(AALTO_SMTP_SERVER)
    smtp.send_message(msg)
    smtp.quit()


def get_job_details():
    pass


def get_ood_url(file_path: str):
    OOD_BASE_URL = "https://ondemand.triton.aalto.fi"
    OOD_DATAROOT = "/pun/sys/dashboard/files/fs"

    return OOD_BASE_URL + OOD_DATAROOT + file_path

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Send email notification for job completion.')
    parser.add_argument('--to', type=str, required=True, help='Email recipient.')
    parser.add_argument('--file_name', type=str, required=True, help='The audio file name to include in the email.')
    parser.add_argument('--file_path', type=str, required=True, help='The file path for creating ondemand url to the result folder.')
    parser.add_argument('--email_subject', type=str, default='Transcription job is completed', help='Email subject')
    parser.add_argument('--sender', type=str, default='no-reply@aalto.fi', help='The sender email address.')
    parser.add_argument('--job_id', type=str, required=False, help='The job ID to include in the email.')
    args = parser.parse_args()

    sendemail(args.to, args.file_name, args.file_path, args.email_subject, args.sender, args.job_id)

if __name__ == '__main__':
    main()
