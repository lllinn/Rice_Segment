import sys
sys.path.append(r'./')  # 将src的上级目录加入sys.path
import os
# from src.utils.email_util import send_email
import smtplib
import ssl
from email.message import EmailMessage
# 引入超参数
import argparse

# subjects = ["模型训练", "模型预测", "数据集制作", "数据集合成"]
# subject = subjects[0]   # 
# body = subject + "完成.."

def send_email(subject, body=None):
    if not body:
        body = subject + "完成.."
    EMAIL_ADDRESS = "zlin_deeplearning@163.com"     # 邮箱的地址
    EMAIL_PASSWORD = "WHTM6eJQPLaSjNUQ"     # 授权码
    
    context = ssl.create_default_context()

    # subjects = ["模型训练", "模型预测", "数据集生成", "数据集合成"]
    # subject = subjects[0]   # 
    # body = subject + "完成.."

    msg = EmailMessage()
    msg['subject'] = subject        # 邮件标题
    msg['From'] = EMAIL_ADDRESS     # 邮件发件人
    msg['To'] = "zlin_deeplearning@163.com"                  # 邮件的收件人
    msg.set_content(body)           # 使用set_content()方法设置邮件的主体内容
    
    # 为了防止忘记关闭连接也可以使用with语句
    with smtplib.SMTP_SSL("smtp.163.com", 465, context=context) as smtp:      # 完成加密通讯
    
        # 连接成功后使用login方法登录自己的邮箱
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    
        smtp.send_message(msg)
import time
folder = r"/root/autodl-tmp/train"
while True:
    file_list = os.listdir(folder)
    print(len(file_list), len(file_list) / 6150)
    time.sleep(1)