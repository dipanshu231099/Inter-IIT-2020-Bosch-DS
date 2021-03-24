# 1 Installation and user manual
This document gives easy steps to install/set-up and to use the web

## 1.1 Installation
You need to have LAMP(Linux,Apache,MySQL,PHP) on your machine. If you need to install them follow these steps
### 1.1.1 Installing Apache, Mysql and php
 >  To install the Apache web server, run the following command in the Linux terminal

- **sudo apt-get update** <br>
It’s a good idea to refresh your local software package database to make sure you are accessing the latest versions.
- **sudo apt-get install apache2**<br>
 You will be prompted for confirmation to proceed for installation.
In some cases you might need to configure firewall settings.

> To install mysql server run following commands
- **sudo apt install mysql-server**<br>
This will install mysql but you still need to make configuration changes.
- **sudo mysql_secure_installation** <br>
You will be taken through series of steps where you can alter mysql installation security options. The first prompt will ask if you’d like to set up the a plugin, which can be used to test the strength of your MySQL password. Irrespective of your choice, you will be prompted to set a password for the MySQL root user. Enter and then confirm password.

> To install PHP
- **sudo apt-get install php libapache2-mod-php php-mcrypt php-mysql**<br>
This will install php along with some helper packages.


- **sudo apt-get install -y php-mysqli**<br>
This will install php-mysqli for database-PHP connection
### 1.1.2  To set up the application
- Clone the repository at the Apache root directory.
- Open your web browser and type localhost/\<your app folder>/webpages

## 1.2 Usage manual
* Homepage of the website<br>
    * Homepage show three options for Login for our three type of users- Students, School, Admin. It also shows a option "Sign up" for signing up a new id for students for using first time.
    * Login as student/School/Admin- The interface let user enter their credentials to log into their accounts.

***

![](https://github.com/dipanshu231099/HP-STSE/blob/ujjwalsoni1707-patch-1/public/images/home_page.png)
***
* Sign up - After entering the email address for otp verification, the account registration page opens where you need to enter details and the otp received on the entered email.
***
![](https://github.com/dipanshu231099/HP-STSE/blob/ujjwalsoni1707-patch-1/public/images/pic1.png)
***
![](https://github.com/dipanshu231099/HP-STSE/blob/ujjwalsoni1707-patch-1/public/images/pic2.png)
***
* Dashboard after signing up for students.
    * If the students has not filled the form yet it will show "Fill application form" button to fill the form
    * Clicking "Fill application form" button will land you to fill the application form. To submit application click Submit Application form button.
***
![](https://github.com/dipanshu231099/HP-STSE/blob/ujjwalsoni1707-patch-1/public/images/pic3.png)

