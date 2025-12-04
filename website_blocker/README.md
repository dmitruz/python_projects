Website Blocker Python
Intro
Hi Guys 👋

This repository consists the code of a simple website blocker project implemented in Python. It can be used to block certain websites during work time to reduce distraction thus improving productivity.

The magic
The magic of this project lies on modifying the host file within your computer that manages how you access the web.

Getting started
Well to get started with this project just clone the repository and edit the host file location depending on the OS your using.

    $-> git clone https://github.com/dmitruz/python_projects/website_blocker/
    $-> cd Website-blocker
    $ Website-blocker ->
Adding sites to block + Editing host files
Now open the app.py and then go to line 4 with variable site_to_block and you can add the sites you would like to block during work time

the script will automaticaly detect your OS and will add the host records to the relevent location

One more thing
You would need to set up the starting working + ending working hours where you would like to be restricted from accessing those websites. To do this go to line last line of our code and edit the hours where by.

time_to_block
<img width="1358" height="498" alt="time_to_block" src="https://github.com/user-attachments/assets/ebd7f2c0-9554-4042-9883-5d28fedac979" />

<img width="938" height="666" alt="site_to_block" src="https://github.com/user-attachments/assets/6f43cf86-d2aa-4097-b04f-3137e77375de" />

<img width="1210" height="622" alt="host_file_location" src="https://github.com/user-attachments/assets/47c7d68c-67b6-4807-a42b-5c3d628d32f4" />



Congratulations
Well done you now have a fully functioning website blocker that you have made yourself to improve your productivity, in Python

Issues
Incase you have any difficulties or issues while trying to run the script you can raise it on the issues

Pull Requests
If you have something to add, I welcome pull requests that improve the project, your helpful contribution will be merged as soon as possible.

Give it a Star ✴️
If you find this repo useful , give it a star
