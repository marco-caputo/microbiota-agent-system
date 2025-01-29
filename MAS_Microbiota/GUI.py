from typing import Any

import pygame

from .Environments.Brain.Brain import Brain
from .Environments.Gut.Gut import Gut
from .Environments.Microbiota.Microbiota import Microbiota
from .Utils import Simulation
from MAS_Microbiota.Environments.Microbiota.Agents import *
from MAS_Microbiota.Environments.Brain.Agents import *
from MAS_Microbiota.Environments.Gut.Agents import *

class GUI:
    def __init__(self, width, height, envs):
        self.background_color = (202, 187, 185)
        self.border_color = (255, 255, 255)
        self.width = width
        self.height = height
        self.envs = envs
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.Font(None, 36)
        self.running = True
        self.gut_context = envs[Gut.NAME].context
        self.brain_context = envs[Brain.NAME].context
        self.microbiota_context = envs[Microbiota.NAME].context
        self.grid_width, self.grid_height = Simulation.params['world.width'], Simulation.params['world.height']
        self.paused = False
        self.button_rects = []
        self.params = Simulation.params

        self.legend = [
            ((255, 60, 40), 'Enterobacteriaceae'), # A bright, intense red with a touch of orange to make it bold and unmistakable for Pathogenic-Sugar.
            ((200, 50, 30), 'Streptococcaceae'), # A very vibrant and deeper red, clearer contrast from Enterobacteriaceae.
            ((255, 125, 60), 'Clostridiaceae'),  # A bright, brilliant orange-brown for Sugar.
            ((200, 110, 50), 'Lactobacillaceae'), # A warm, golden brown with more brightness to separate it from Clostridiaceae.
            ((120, 150, 60), 'Prevotellaceae'), # A vibrant olive green to make it stand out more from Lactobacillaceae.
            ((95, 130, 50), 'Bifidobacteriaceae'), # A lighter and more vivid olive-brown for clearer distinction from Prevotellaceae.
            ((95, 180, 95), 'Ruminococcaceae'),  # A bright, rich green with brown undertones for Carb-Fiber.
            ((70, 160, 70), 'Lachnospiraceae'), # A cool and intense forest green, different enough from Ruminococcaceae.

            ((255, 220, 180), 'Sugar'), # A light, brilliant peachy color, making it brighter and more clearly different from brown tones.
            ((250, 250, 150), 'Carbohydrate'), # A vivid, sunny yellow for better contrast against Sugar and other colors.
            ((220, 255, 180), 'Fiber'), # A fresh, light green with high brightness, making it clearer than other greens.
            ((200, 250, 255), 'SCFA'),  # A cool, bright cyan-blue tone for clear contrast against the warmer colors.
            ((255, 180, 255), 'Precursor'),  # A soft yet more saturated pinkish-lavender tone for better clarity.
            ((100, 240, 255), 'Serotonin'),  # A bright, clear cyan to make it stand out with a pure, fresh cyan hue. 'Serotonin'), # A cool, vibrant purple-cyan mix, leaning towards cyan to keep it fresh and distinct.
            ((100, 160, 255), 'Dopamine'), # A richer purple-yellow mix, more saturated to distinguish it from serotonin.
            ((100, 70, 255), 'Norepinephrine'), # A much deeper purple-blue with a touch of blue, making it darker and cooler for better contrast.

            ((147, 112, 219), 'Active AEP'),  # Soft purple
            ((128, 0, 255), 'Hyperactive AEP'),  # Purple-blue
            ((173, 216, 230), 'Protein'),  # Light neutral cyan
            ((100, 149, 237), 'Cleaved Protein'),  # Deep blue
            ((139, 0, 0), 'Oligomer'),  # Dark red
            ((220, 220, 220), 'External Input'),  # Neutral gray
            ((240, 240, 240), 'Treatment'),  # Light gray

            ((144, 238, 144), 'Resting Microglia'),  # Soft green
            ((34, 139, 34), 'Active Microglia'),  # Deep green
            ((255, 105, 180), 'Healthy Neuron'),  # Bright pink
            ((255, 69, 0), 'Damaged Neuron'),  # Vibrant orange-red
            ((0, 0, 0), 'Dead Neuron'),  # Black
            ((255, 160, 0), 'Pro-inflammatory Cytokine'),  # Golden yellow
            ((255, 255, 102), 'Anti-inflammatory Cytokine')  # Light yellow
        ]

        self.color_dict = {}
        for label in ['Microglia', 'Neuron', 'Cytokine', 'AEP', 'Neurotransmitter', 'Substrate']:
            self.color_dict[label] = {}
        for color, label in self.legend:
            if label == 'Serotonin': self.color_dict['Neurotransmitter'][NeurotransmitterType.SEROTONIN] = color
            elif label == 'Dopamine': self.color_dict['Neurotransmitter'][NeurotransmitterType.DOPAMINE] = color
            elif label == 'Norepinephrine': self.color_dict['Neurotransmitter'][NeurotransmitterType.NOREPINEPHRINE] = color
            elif label == 'Active AEP': self.color_dict['AEP'][AEPState.ACTIVE] = color
            elif label == 'Hyperactive AEP': self.color_dict['AEP'][AEPState.HYPERACTIVE] = color
            elif label == 'Cleaved Protein': self.color_dict['CleavedProtein'] = color
            elif label == 'External Input': self.color_dict['ExternalInput'] = color
            elif label == 'Resting Microglia': self.color_dict['Microglia'][MicrogliaState.RESTING] = color
            elif label == 'Active Microglia': self.color_dict['Microglia'][MicrogliaState.ACTIVE] = color
            elif label == 'Healthy Neuron': self.color_dict['Neuron'][NeuronState.HEALTHY] = color
            elif label == 'Damaged Neuron': self.color_dict['Neuron'][NeuronState.DAMAGED] = color
            elif label == 'Dead Neuron': self.color_dict['Neuron'][NeuronState.DEAD] = color
            elif label == 'Pro-inflammatory Cytokine': self.color_dict['Cytokine'][CytokineState.PRO_INFLAMMATORY] = color
            elif label == 'Anti-inflammatory Cytokine': self.color_dict['Cytokine'][CytokineState.NON_INFLAMMATORY] = color
            elif label == 'Fiber': self.color_dict['Substrate'][SubstrateType.FIBER] = color
            elif label == 'Carbohydrate': self.color_dict['Substrate'][SubstrateType.CARBOHYDRATE] = color
            elif label == 'Sugar': self.color_dict['Substrate'][SubstrateType.SUGAR] = color
            else: self.color_dict[label] = color


    # Function to update the interface
    def pygame_update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # If the 'X' button is clicked, stop the simulation
                print("Ending the simulation.")
                Simulation.model.at_end()
                Simulation.model.comm.Abort()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_button_click(event.pos)

        while self.paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # If the 'X' button is clicked, stop the simulation
                    print("Ending the simulation.")
                    Simulation.model.at_end()
                    Simulation.model.comm.Abort()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_button_click(event.pos)

        # Updates the Pygame GUI based on the current state of the Repast simulation
        self.update()
        pygame.display.flip()

    # Function to update the screen after each tick
    def update(self):
        # Update contexts
        self.gut_context = self.envs[Gut.NAME].context
        self.brain_context = self.envs[Brain.NAME].context
        self.microbiota_context = self.envs[Microbiota.NAME].context

        # Fill background and draw border rectangle
        self.screen.fill(self.background_color)
        inner_rect = (50, 50, self.width - 100, self.height - 300)
        pygame.draw.rect(self.screen, self.border_color, inner_rect)

        # Draw section titles
        text_y_position = inner_rect[1] - 30
        self._draw_centered_text("Microbiota Environment", self.width // 6, text_y_position)
        self._draw_centered_text("Gut Environment", self.width // 2, text_y_position)
        self._draw_centered_text("Brain Environment", 5 * self.width // 6, text_y_position)

        # Draw separating line
        pygame.draw.line(self.screen, (0, 0, 0),
                         (self.width // 3, inner_rect[1]),
                         (self.width // 3, inner_rect[1] + inner_rect[3]), 4)
        pygame.draw.line(self.screen, (0, 0, 0),
                         (2 * self.width // 3, inner_rect[1]),
                         (2 * self.width // 3, inner_rect[1] + inner_rect[3]), 4)
        # Draw buttons and legend
        self.draw_buttons()
        self.draw_legend()

        # Define areas and draw agents
        microbiota_area = (50, 50, self.width // 3 - 47, self.height - 300)
        gut_area = (self.width // 3 + 3, 50, self.width // 3 - 3, self.height - 300)
        brain_area = (2 * self.width // 3 + 3, 50, self.width // 3 - 50, self.height - 300)
        self._draw_context_agents(self.microbiota_context, microbiota_area)
        self._draw_context_agents(self.gut_context, gut_area)
        self._draw_context_agents(self.brain_context, brain_area)

    def _draw_centered_text(self, text, x_center, y):
        rendered_text = self.font.render(text, True, (0, 0, 0))
        text_x_position = x_center - rendered_text.get_width() // 2
        self.screen.blit(rendered_text, (text_x_position, y))

    def _draw_context_agents(self, context, area):
        self.draw_agents(context.agents(), area)

    # Function to draw agents on the screen
    def draw_agents(self, agents, area):
        radius = 5
        
        for agent in agents:
            if ((isinstance(agent, Bacterium) and Simulation.params['agents_display'][agent.context]["Bacterium"]) or
                (not isinstance(agent, Bacterium) and Simulation.params['agents_display'][agent.context][agent.__class__.__name__])):
                x_center = area[0] + (agent.pt.x / self.grid_width) * area[2]
                y_center = area[1] + (agent.pt.y / self.grid_height) * area[3]

                # Adjust x and y to keep the entire circle within the area
                x = max(area[0] + radius, min(x_center, area[0] + area[2] - radius))
                y = max(area[1] + radius, min(y_center, area[1] + area[3] - radius))

                color = self.get_agent_color(agent)
                pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)

    # Function to get the color of an agent based on its type and state
    def get_agent_color(self, agent):
        class_name = agent.__class__.__name__
        if isinstance(self.color_dict[class_name], dict):
            if isinstance(agent, Neurotransmitter):
                return self.color_dict[class_name][agent.neurotrans_type]
            elif isinstance(agent, Substrate):
                return self.color_dict[class_name][agent.sub_type]
            else:
                return self.color_dict[class_name][agent.state]
        else:
            return self.color_dict[class_name]

    # Function to draw the legend on the screen
    def draw_legend(self):
        # Define legend position and size
        legend_x = 60
        legend_y = self.height - 230
        legend_radius = 6
        legend_spacing = 35
        text_offset_x = 15
        row_spacing = 20

        # Define legend title
        legend_title_font = pygame.font.Font(None, 30)
        legend_title_surface = legend_title_font.render("Legend:", True, (0, 0, 0))
        title_width, title_height = legend_title_surface.get_size()

        # Define legend fonts
        legend_font = pygame.font.Font(None, 20)
        legend_color = (0, 0, 0)  # Black

        # Calculate number of items per column
        items_per_column = [8, 8, 7, 7]

        # Calculate legend background size
        legend_width = 400
        legend_height = max(items_per_column) * row_spacing + 70

        # Calculate legend title position
        title_x = legend_x + (legend_width - title_width)
        title_y = legend_y + 5

        # Draw legend title
        self.screen.blit(legend_title_surface, (title_x, title_y))

        # Draw legend items
        item = 0
        for column in range(len(items_per_column)):
            for row in range(items_per_column[column]):
                color, label = self.legend[item]
                circle_x = legend_x + column * (legend_width // 2) + legend_radius
                circle_y = legend_y + 35 + row * row_spacing + legend_radius
                pygame.draw.circle(self.screen, color, (circle_x, circle_y), legend_radius)
                legend_text_surface = legend_font.render(label, True, legend_color)
                text_width, text_height = legend_text_surface.get_size()
                text_x = legend_x + column * (legend_width // 2) + 2 * legend_radius + text_offset_x
                text_y = circle_y - text_height // 2
                self.screen.blit(legend_text_surface, (text_x, text_y))
                item += 1

    # Function to draw the play and stop buttons on the screen
    def draw_buttons(self):
        button_font = pygame.font.Font(None, 30)
        buttons = ["Play", "Stop"]
        button_width = 120
        button_height = 40
        button_spacing = 10

        total_width = (button_width * len(buttons)) + (button_spacing * (len(buttons) - 1))
        start_x = (self.width - total_width) // 4 * 3

        for i, button_text in enumerate(buttons):
            button_x = start_x + i * (button_width + button_spacing)
            button_rect = pygame.Rect(button_x, self.height - 150, button_width, button_height)
            pygame.draw.rect(self.screen, (137, 106, 103), button_rect)
            button_surface = button_font.render(button_text, True, (0, 0, 0))
            # Center the text on the button
            button_text_rect = button_surface.get_rect(center=button_rect.center)
            self.screen.blit(button_surface, button_text_rect.topleft)

            self.button_rects.append((button_rect, button_text))

    # Function to handle mouse clicks on the buttons
    def handle_button_click(self, mouse_pos):
        for button_rect, button_text in self.button_rects:
            if button_rect.collidepoint(mouse_pos):
                self.on_button_click(button_text)

    # Function to handle button clicks
    def on_button_click(self, button_text):
        if button_text == "Play":
            self.paused = False
        elif button_text == "Stop":
            self.paused = True