import pygame
import numpy
import tensorflow as tf
import os
import cv2
import threading

#load model
model = tf.keras.models.load_model(os.path.join("digitsRecognizer","digits.model"), compile=True)


#initialize window
pygame.init()
pygame.display.set_caption("Cool Recognizer")
WIDTH, HEIGHT = 500, 500
win = pygame.display.set_mode((WIDTH, HEIGHT))

#constants for drawing sufrace
global paint_surface
SURFACE_WIDTH = 28 * 12
SURFACE_HEIGHT = 28 * 12
scale_rect = pygame.Rect(((WIDTH-SURFACE_WIDTH)/2,(HEIGHT-SURFACE_HEIGHT)/1.3, SURFACE_WIDTH, SURFACE_HEIGHT))

#colors
WHITE = (255,255,255)
BLACK = (0,0,0)
# RED = (255,0,0)
TEXT = pygame.font.SysFont("arial", 15)


def init_paint_surface():
    """
    Initialize surface to draw numbers
    """
    global paint_surface, scale_rect
    surface = pygame.Rect((WIDTH-SURFACE_WIDTH)/2,(HEIGHT-SURFACE_HEIGHT)/1.3, SURFACE_WIDTH, SURFACE_HEIGHT)
    paint_surface = win.subsurface(surface)
    paint_surface.fill(BLACK)
    pygame.draw.rect(paint_surface, BLACK, scale_rect)


def draw(cursor_x, cursor_y):
    """Creates a circle at cursor place in paint surface

    Args:
        cursor_x
        cursor_y
    """
    pygame.draw.circle(paint_surface, WHITE, [cursor_x, cursor_y], 8)
    pygame.display.update()


def recognize():
    """gets array of pixels and passes to model
    """
    view = pygame.surfarray.array3d(paint_surface) #gets width, height, channel
    view = view.transpose([1, 0, 2]) #transpose from width, height, channel to  height, width, channel
    # print(view)
    img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR) #converts from rgb to bgr
    # cv2.imshow('windowname',img_bgr[:,:,0])
    # print(img_bgr[:,:,0].shape)
    resized_image = cv2.resize(img_bgr, (28, 28))  #resizes to 28, 28
    predict = model.predict(numpy.expand_dims(resized_image[:,:,0],0)) # before passing to model creating extra dimension and choosing 0 channel
    # print(numpy.argmax(predict))
    probability(predict)


def probability(predict):
    surface = pygame.Rect(0, 50, 50, 350)
    prob_surface = win.subsurface(surface)
    prob_surface.fill(WHITE)
    # for key, value in enumerate(predict[0]):
    #     print(f"Chance of {key} is {value*100}%")
    for key, value in enumerate(predict[0]):
        text_surface = TEXT.render(f"{key}: {value*100}%", False, BLACK)
        prob_surface.blit(text_surface, (0, 50+(key*20)))
        # print(TEXT.size("7: 100%")) 45:18
        pygame.display.update()


def reset_paint_surface():
    paint_surface.fill(BLACK)


def main():
    FPS = 300
    clock = pygame.time.Clock()
    running = True
    
    win.fill(WHITE)
    init_paint_surface()
    
    while running:
        clock.tick(FPS)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                break
            if pygame.mouse.get_pressed()[0]:
                cursor_x, cursor_y = pygame.mouse.get_pos()
                draw(cursor_x-scale_rect.x,cursor_y-scale_rect.y)
                t1 = threading.Thread(target=recognize())
                t1.start()
                # recognize()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    reset_paint_surface()
                if event.key == pygame.K_RETURN:
                    recognize()

if __name__ == '__main__':
    main()