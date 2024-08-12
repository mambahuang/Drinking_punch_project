import pygame

pygame.mixer.init()

def play_lose():
    pygame.mixer.music.load('/jetson-beer/user_interface/media/audio/ahh.wav')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def play_win():
    pygame.mixer.music.load('/jetson-beer/user_interface/media/audio/yeah.wav')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

