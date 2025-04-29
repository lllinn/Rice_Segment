# 使用 smtplib 模块发送纯文本邮件
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate volume of a cylinder')
    # 给这个解析对象添加命令行参数
    parser.add_argument('-t', '--topic', default="命令运行",type=str, help='topic of email')
    args = parser.parse_args()  # 获取所有参数
    topic = args.topic
    content = topic + "完成.."
    send_email(topic, content)

