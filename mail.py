from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from smtplib import SMTP
from email.mime.application import MIMEApplication
import smtplib
from credentials import CREDENTIALS

# https://stackoverflow.com/questions/63911614/how-do-i-send-an-email-to-my-gmail-using-python


def test_conn_open(conn):
    try:
        status = conn.noop()[0]
    except:  # smtplib.SMTPServerDisconnected
        status = -1
    return True if status == 250 else False


from_address = CREDENTIALS['from_address']
to_address = CREDENTIALS['to_address']


def send_correo(file_name):
    message = MIMEMultipart()
    message['Subject'] = "Reporte drone"
    text = MIMEText(
        "Ya acabó!")
    message.attach(text)

    directory = file_name
    with open(directory, 'rb') as opened:
        openedfile = opened.read()
    attachedfile = MIMEApplication(openedfile, _subtype="png")
    attachedfile.add_header('content-disposition', 'attachment', filename=file_name)
    message.attach(attachedfile)

    smtp = SMTP("smtp.live.com", 587)
    i = 0
    while i <= 3:
        if not test_conn_open(smtp):
            smtp = SMTP("smtp.gmail.com", 465)
            i += 1
        else:
            break
    smtp.starttls()
    smtp.login(CREDENTIALS['from_address'], CREDENTIALS['password'])
    smtp.sendmail(from_address, to_address, message.as_string())
    smtp.quit()