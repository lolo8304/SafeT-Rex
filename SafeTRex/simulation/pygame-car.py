import os
import pygame
from math import tan, radians, degrees, copysign
from pygame.math import Vector2

from pygame import font as pygame_font
from pygame import Surface, Rect

class CarStructure:
    def __init__(self, length, width, centralPoint, rear_axle_distance, front_axle_distance, tire_radius, tire_width):
        self.H = length
        self.length = length
        self.L = self.H - rear_axle_distance - front_axle_distance
        self.g2 = rear_axle_distance
        self.g = front_axle_distance
        self.T = tire_width
        self.T2 = tire_radius * 2
        self.W2 = width
        self.W =  self.W2 - tire_width
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "car.png")
        self.car_image = self.surface(pygame.image.load(image_path))
        self.central_point = centralPoint
        self.central_point_factor = Vector2(centralPoint.x / length, centralPoint.y / width)
        print(self.central_point_factor)

    # base simulation image reference: W=128, H=64, Length=4
    def surface(self, image):
        image_rect = image.get_rect()
        len_factor = self.length / 4.0
        hw_factor = 64.0 / 128.0
        size_real_rect = Rect(0, 0, image_rect.w * len_factor, image_rect.w * len_factor * hw_factor)
        return pygame.transform.smoothscale(image, (size_real_rect.w, size_real_rect.h))


    def draw(self, central_point, angle, length):
        pass

class Car:
    def __init__(self, x, y, car_structure, angle=0.0, max_steering=30, max_acceleration=5.0):
        self.structure = car_structure
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)
        self.angle = angle
        self.length = self.structure.length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_velocity = 20
        self.brake_deceleration = 10
        self.free_deceleration = 2
        self.keep_speed = True
        self.angular_velocity = 0

        self.car_image = self.structure.car_image

        self.acceleration = 0.0
        self.steering = 0.0
        self.center_path = []

    def update(self, dt):
        self.velocity += (self.acceleration * dt, 0)
        self.velocity.x = max(-self.max_velocity, min(self.velocity.x, self.max_velocity))

        if self.steering:
            self.turning_radius = self.length / tan(radians(self.steering))
            self.angular_velocity = self.velocity.x / self.turning_radius
        else:
            self.angular_velocity = 0
            self.turning_radius = 0

        self.position += self.velocity.rotate(-self.angle) * dt
        self.angle += degrees(self.angular_velocity) * dt
        self.angle = self.angle % 360


    def minmax_acceleration(self):
        self.acceleration = max(-self.max_acceleration, min(self.acceleration, self.max_acceleration))

    def minmax_steering(self):
        self.steering = max(-self.max_steering, min(self.steering, self.max_steering))



    def speed_inc(self, dt):
        if self.velocity.x < 0:
            self.acceleration = self.brake_deceleration
        else:
            self.acceleration += 1 * dt
        self.minmax_acceleration()

    def speed_dec(self, dt):
        if self.velocity.x > 0:
            self.acceleration = -self.brake_deceleration
        else:
            self.acceleration -= 1 * dt
        self.minmax_acceleration()

    def speed_break(self, dt):
        if abs(self.velocity.x) > dt * self.brake_deceleration:
            self.acceleration = -copysign(self.brake_deceleration, self.velocity.x)
        else:
            self.acceleration = -self.velocity.x / dt
        self.minmax_acceleration()

    def speed_slowdown(self, dt):
        if self.keep_speed:
            self.acceleration = 0
        else:
            if abs(self.velocity.x) > dt * self.free_deceleration:
                self.acceleration = -copysign(self.free_deceleration, self.velocity.x)
            else:
                if dt != 0:
                    self.acceleration = -self.velocity.x / dt
            self.minmax_acceleration()

    def left(self, dt):
        self.steering += 30 * dt
        self.minmax_steering()

    def right(self, dt):
        self.steering -= 30 * dt
        self.minmax_steering()

    def straight(self, dt):
        self.steering = 0

    def draw0(self, surface, ppu):
        rotated = pygame.transform.rotate(self.car_image, self.angle)
        rotate = self.structure.central_point.rotate(self.angle)
        rect = rotated.get_rect()

        self.addpath(self.position * ppu)
        surface.blit(rotated, self.position * ppu - (rect.w / 2, rect.h / 2))
        #self.addpath(self.position * ppu)
        #surface.blit(rotated, self.position * ppu - (rotate.x, rotate.y))

    def draw1(self, surface, ppu):
        image = self.car_image.copy()
        pygame.draw.rect(image, (255,0,0), image.get_rect(), 1)

        rotated = pygame.transform.rotate(image, self.angle)
        rotate = self.structure.central_point.rotate(self.angle)
        rect = rotated.get_rect()
        self.addpath(self.position * ppu)
        factor = self.structure.central_point_factor
        surface.blit(rotated, self.position * ppu - (rotate.x * ppu, rotate.y * ppu))
    
    def addpath(self, pos):
        if (len(self.center_path) > 0):
            if self.center_path[-1] == pos:
                pass
            else:
                self.center_path.append(pos)
            if (len(self.center_path) > 500):
                del self.center_path[0]
        else:
            self.center_path.append(pos)

    def draw_center_path(self, surface, ppu):
        for pos in self.center_path:
            pygame.draw.circle(surface, (255,0,0), (int(pos.x), int(pos.y)), 1, 1)


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Car tutorial")
        width = 640 * 3 // 2
        height = 720
        self.screen = pygame.display.set_mode((width, height))
        self.f = pygame_font.SysFont("consolas", 20)
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False

    def show_dashboard_str(self, text, offset):
        s = self.f.render(text, True, (255, 255, 255))
        self.screen.blit(s, (0, offset))
        return offset + self.f.get_linesize()+2

    def show_dashboardentry_number(self, title, value, offset):
        text = '{:25.25}'.format(title)+": "+'{:6.2f}'.format(value)
        return self.show_dashboard_str(text, offset)

    def show_dashboardentry_str(self, title, value, offset):
        text = '{:25.25}'.format(title)+": "+value
        return self.show_dashboard_str(text, offset)

    def show_dashboard(self, car):
        offset = 0
        offset = self.show_dashboardentry_str("position", str(car.position), offset)
        offset = self.show_dashboardentry_number("size", car.angular_velocity, offset)
        offset = self.show_dashboardentry_number("steering", car.steering, offset)
        offset = self.show_dashboardentry_number("acceleration", car.acceleration, offset)
        offset = self.show_dashboardentry_number("velocity", car.velocity.x, offset)
        offset = self.show_dashboardentry_number("angle", car.angle, offset)
        offset = self.show_dashboardentry_number("angular velocity", car.angular_velocity, offset)
        offset = self.show_dashboardentry_number("turning radius", car.turning_radius, offset)


    def run(self):
        car_structure = CarStructure(4, 2, Vector2(2,1), 0, 0, 0, 0)
        car = Car(10.0,10.0, car_structure)
        ppu = 32

        while not self.exit:
            dt = self.clock.get_time() / 1000

            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            # User input
            pressed = pygame.key.get_pressed()

            if pressed[pygame.K_q]:
              self.exit = True

            if pressed[pygame.K_UP]:
                car.speed_inc(dt)
            elif pressed[pygame.K_DOWN]:
                car.speed_dec(dt)
            elif pressed[pygame.K_SPACE]:
                car.speed_break(dt)
            else:
                car.speed_slowdown(dt)

            if pressed[pygame.K_RIGHT]:
                car.right(dt)
            elif pressed[pygame.K_LEFT]:
                car.left(dt)
            else:
                pass
                #car.straight(dt)

            # Logic
            car.update(dt)

            # Drawing
            self.screen.fill((0, 0, 0))
            self.show_dashboard(car)
            #car.draw(self.screen, ppu)
            car.draw1(self.screen, ppu)
            car.draw_center_path(self.screen, ppu)

            pygame.display.flip()

            self.clock.tick(self.ticks)
        pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.run()