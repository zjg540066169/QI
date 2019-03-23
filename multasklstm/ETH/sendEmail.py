# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:39:36 2018

@author: wang
"""

from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr,formataddr
import smtplib

def sendMail(title,string):
    def jiaformatadd(s):
        name,addr = parseaddr(s)
        return formataddr((Header(name,'UTF-8').encode(),addr))
    #发送邮件账号 PS：记得开SMTP 
    from_addr ='jungang.zou@gmail.com'
    #密码
    password ='540066169zjg?!'
    #接受邮件账号
    to_addr = 'linfan@shareted.com'
    cc_addr = "cj5260@163.com"
    #to_addr = '540066169@qq.com'
    #发送服务器
    smtup_server='smtp.gmail.com'
    #邮件内容  MIMEtext中是可以输入html代码的
    msg = MIMEText(string)
    msg['From'] = jiaformatadd('邹俊岗<%s>'%from_addr)
    msg['TO'] = jiaformatadd('Fan Lin<%s>'%to_addr)
    msg['Cc'] = jiaformatadd('Jian Cai<%s>'%cc_addr)
    msg['Subject'] = Header(title,"UTF-8").encode()
    #发送邮件  一般服务器默认25端口
    server = smtplib.SMTP(smtup_server,25)
    server.set_debuglevel(1)
    server.starttls()
    server.login(from_addr,password)
    
    server.sendmail(from_addr,[to_addr],msg.as_string())
    server.quit()

if __name__=='__main__':
    pass