import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

from pathlib import Path, PosixPath

AALTO_SMTP_SERVER = 'smtp.aalto.fi'
RSE_EAMIL = 'rse-group@aalto.fi'

def sendemail(to: str, 
              file_name: PosixPath,
              file_path: str,
              subject: str,
              sender: str,
              send_attachments: bool,
              job_id: int = None):

    msg = MIMEMultipart('alternative')
   
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = to

    attachments: list[MIMEApplication]

    # Job successful
    if not job_id:
        body = f"""
        <html>
            <body>
                <p>Hi,</p>
                <p>Your transcription job for file '{file_name}' is now completed.</p>
                <p>Transcripted files are available inside the <a href="{get_ood_url(file_path)}">results folder</a>.</p>"""
        
        if send_attachments:
            # Remove the last </p>
            body = body[:-4] + f"""
             Transcripted results are also attached to this email.</p>
            """
            attachments = get_result_files(file_name, file_path)

        body += f"""
                <p>If you any questions or feedbacks, please reply to this email or visit our <a href="https://scicomp.aalto.fi/help/garage/">daily garage</a>, every day at 13:00 EET.</p>
                <p>Best,</p>
                <p>Aalto Scientific Computing</p>
            </body>
        </html>
        """    
        
        msg.attach(MIMEText(body, 'html'))    

    # Job failed
    else:
        body = f"""
        <html>
            <body>
                <p>Hi,</p>
                <p>Your transcription job #{job_id} for file '{file_name}' has been failed :(</p>
                <p>Log files are available inside the <a href="{get_ood_url(file_path)}">log folder</a> and are also attached to this email.</p>
                <p>Please reply to this email so our team can investigate the issue or visit our <a href="https://scicomp.aalto.fi/help/garage/">daily garage</a>, every day at 13:00 EET.</p>
                <p>Best,</p>
                <p>Aalto Scientific Computing</p>
            </body>
        </html>
        """

        msg.attach(MIMEText(body, 'html'))

        attachments = get_log_files(file_name, file_path, job_id)
    
    if send_attachments:
        for attachment in attachments:
            msg.attach(attachment)
    
    try:
        smtp = smtplib.SMTP(AALTO_SMTP_SERVER)
        smtp.send_message(msg)
    except smtplib.SMTPException as e:
        print(f"Failed to send email. Error {e}")
    finally:
        smtp.quit()


def get_result_files(file_name: PosixPath, file_path:str):
    txt_file = f"{file_path}/{Path(file_name).stem}.txt"
    csv_file = f"{file_path}/{Path(file_name).stem}.csv"

    res = []

    with open(txt_file, 'rb') as f:
        log = MIMEApplication(f.read(), Name=f"{file_name}.txt")
        log['Content-Disposition'] = f'attachment; filename="{file_name}.txt"'
        res.append(log)

    with open(csv_file, 'rb') as f:
        log = MIMEApplication(f.read(), Name=f"{file_name}.csv")
        log['Content-Disposition'] = f'attachment; filename="{file_name}.csv"'
        res.append(log)

    return res

def get_ood_url(file_path: str):
    OOD_BASE_URL = "https://ondemand.triton.aalto.fi"
    OOD_DATAROOT = "/pun/sys/dashboard/files/fs"

    return OOD_BASE_URL + OOD_DATAROOT + file_path


def get_log_files(file_name: PosixPath, file_path:str, job_id: int):
    error_file = f"{file_path}/speech2text_{file_name}_{job_id}.err"
    out_file = f"{file_path}/speech2text_{file_name}_{job_id}.out"

    res = []

    with open(error_file, 'rb') as f:
        log = MIMEApplication(f.read(), Name=f"{file_name}_{job_id}.err")
        log['Content-Disposition'] = f'attachment; filename="{file_name}_{job_id}.err"'
        res.append(log)

    with open(out_file, 'rb') as f:
        log = MIMEApplication(f.read(), Name=f"{file_name}_{job_id}.out")
        log['Content-Disposition'] = f'attachment; filename="{file_name}_{job_id}.out"'
        res.append(log)

    return res


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Send email notification for job completion.')
    parser.add_argument('--to', type=str, required=True, help='Email recipient.')
    parser.add_argument('--file_name', type=str, required=True, help='The audio file name to include in the email.')
    parser.add_argument('--file_path', type=str, required=True, help='The file path for creating ondemand url to the result folder.')
    parser.add_argument('--email_subject', type=str, required=True, help='Email subject')
    parser.add_argument('--sender', type=str, default=RSE_EAMIL, help='The sender email address.')
    parser.add_argument('--attachment', type=bool, default=False, help='Send results via email.')
    parser.add_argument('--job_id', type=str, required=False, help='The job ID to include in the email.')
    args = parser.parse_args()

    sendemail(args.to, args.file_name, args.file_path, args.email_subject, args.sender, args.attachment, args.job_id)

if __name__ == '__main__':
    main()
