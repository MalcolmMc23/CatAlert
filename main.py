import pygame
def play_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("cat_alert.mp3")
    pygame.mixer.music.play()

play_sound()
