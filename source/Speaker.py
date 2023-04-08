from email import message
import random


class Speaker:
    def __init__(self, name):
        self.name = name
        red = random.randint(0,255)
        green = random.randint(0,255)
        blue = random.randint(0,255)
        self.message = ""
        self.message_status = True
        self.text_color = [red,green,blue]
    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name
    def get_text_color(self):
        return self.text_color
    def set_text_color(self, text_color):
        self.text_color = text_color
