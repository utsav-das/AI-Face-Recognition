from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.widget import Widget

# Adjust screen size (optional)
Window.size = (360, 640)
Window.clearcolor = (1, 1, 1, 1)  # white background

# Colors
DARK_BLUE = (0/255, 51/255, 102/255, 1)  # RGB for dark blue
WHITE = (1, 1, 1, 1)

# Placeholder screen for home
class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)

        # Spacer to push buttons to vertical center
        layout.add_widget(Widget(size_hint_y=None, height=100))

        btn_faceid = self.create_button('Face Recognition', self.goto_faceid)
        btn_capture = self.create_button('New Profile Creation', self.goto_capture)
        btn_exit = self.create_button('Exit App', lambda x: App.get_running_app().stop())

        layout.add_widget(btn_faceid)
        layout.add_widget(btn_capture)
        layout.add_widget(btn_exit)

        # Spacer at bottom for balance
        layout.add_widget(Widget(size_hint_y=None, height=100))

        self.add_widget(layout)

    def create_button(self, text, callback):
        btn = Button(
            text=text,
            size_hint=(1, None),
            height=80,
            background_normal='',
            background_color=DARK_BLUE,
            color=WHITE,
            font_name='Calibri',
            font_size='20sp'
        )
        btn.bind(on_press=callback)
        return btn

    def goto_faceid(self, instance):
        self.manager.current = 'faceid'
        Clock.schedule_once(lambda dt: self.manager.get_screen('faceid').start_app(), 0)

    def goto_capture(self, instance):
        self.manager.current = 'capture'
        Clock.schedule_once(lambda dt: self.manager.get_screen('capture').start_app(), 0)

# Wrapper for faceid2.py
class FaceIDScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loaded = False

    def start_app(self):
        if not self.loaded:
            from faceid2 import CamApp  # import here to avoid circular import
            self.faceid_app = CamApp()
            self.faceid_app.root = self.faceid_app.build()

            back_btn = Button(
                text="← Back to Home",
                size_hint=(1, 0.1),
                background_normal='',
                background_color=DARK_BLUE,
                color=WHITE,
                font_name='Calibri',
                font_size='18sp'
            )
            back_btn.bind(on_press=self.go_home)

            layout = BoxLayout(orientation='vertical')
            layout.add_widget(back_btn)
            layout.add_widget(self.faceid_app.root)

            self.add_widget(layout)
            self.loaded = True

    def go_home(self, instance):
        self.clear_widgets()
        self.loaded = False
        self.manager.current = 'home'

# Wrapper for capture_face.py
class CaptureScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loaded = False

    def start_app(self):
        if not self.loaded:
            from capture_face import FaceCaptureApp  # import here to avoid circular import
            self.capture_app = FaceCaptureApp()
            self.capture_app.root = self.capture_app.build()

            back_btn = Button(
                text="← Back to Home",
                size_hint=(1, 0.1),
                background_normal='',
                background_color=DARK_BLUE,
                color=WHITE,
                font_name='Calibri',
                font_size='18sp'
            )
            back_btn.bind(on_press=self.go_home)

            layout = BoxLayout(orientation='vertical')
            layout.add_widget(back_btn)
            layout.add_widget(self.capture_app.root)

            self.add_widget(layout)
            self.loaded = True

    def go_home(self, instance):
        if hasattr(self.capture_app, 'capture'):
            self.capture_app.capture.release()
        self.clear_widgets()
        self.loaded = False
        self.manager.current = 'home'

# Main App
class MainApp(App):
    def build(self):
        sm = ScreenManager(transition=FadeTransition())
        sm.add_widget(HomeScreen(name='home'))
        sm.add_widget(FaceIDScreen(name='faceid'))
        sm.add_widget(CaptureScreen(name='capture'))
        return sm

if __name__ == '__main__':
    MainApp().run()
