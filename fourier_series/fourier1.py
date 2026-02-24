from manim import *
import numpy as np
import math

class heat_equation(ThreeDScene):
    def construct(self):

        assi = ThreeDAxes(
            x_range=[0, 4.5, 1],
            y_range=[0, 2, 1],
            z_range=[-5, 5, 1],
            axis_config={"color": WHITE},
        ).scale(0.6)

        assi1 = assi.copy().move_to(2* OUT).rotate(-10*DEGREES, axis = LEFT).scale(0.5)
        assi2 = assi1.copy().shift(LEFT*4.1).rotate(10* DEGREES, axis = OUT)
        assi3 = assi1.copy().shift(RIGHT*4.4).rotate(-11*DEGREES, axis = OUT)


        x_label = always_redraw(lambda: assi.get_x_axis_label("x",  buff=0.1).scale(0.7).rotate(PI/2, axis = RIGHT))
        y_label = always_redraw(lambda: assi.get_y_axis_label("t", buff=0.1, rotation = 0).scale(0.7))
        z_label = always_redraw(lambda: assi.get_z_axis_label("T(x,t)",  buff=0.1).scale(0.7))

        axis_labels = VGroup(x_label, y_label, z_label)
        
        
        N_TERMS = 4
        L = 2

        def temperature(x, t_val):
            total = 0
            for n in range(1, N_TERMS + 1):
                A_n = 1 / n
                term = 2*A_n * np.sin(n *PI* x / L) * np.exp(-(n *PI/ L)**2 * t_val)
                total += term
            return total

        t = ValueTracker(0)

        
        slicer1 = ParametricFunction(lambda u:assi2.c2p(
            u,0,2* np.sin(1*PI*u/ L)),
            t_range= [0,4]).set_color_by_gradient([BLUE, RED]) 
        
        slicer2 = ParametricFunction(lambda u:assi1.c2p(
            u,0,np.sin(2*PI*u/ L)),
            t_range= [0,4]).set_color_by_gradient([BLUE, RED])  
        
        slicer3 = ParametricFunction(lambda u:assi3.c2p(
            u,0,2/3 * np.sin(3*PI*u/ L)),
            t_range= [0,4]).set_color_by_gradient([BLUE, RED])  
        
        
        slicers = VGroup(slicer1, slicer2, slicer3)
        
        
        self.set_camera_orientation(theta= -90 * DEGREES, phi = 90* DEGREES)
        self.wait()
        self.play(Create(assi), Create(axis_labels))
        self.wait()    
        self.play(assi.animate.move_to(-1.7* OUT).scale(0.5))
        self.wait(1)
        self.play(Create(assi1), Create(assi2), Create(assi3))
        self.wait()
        self.play(Create(slicers), run_time=3)
        
        slicer = always_redraw(lambda : ParametricFunction(lambda u: assi.c2p(
            u, t.get_value(), temperature(u,t.get_value())),
            t_range = [0, 4]).set_color_by_gradient([BLUE, RED]))
        
        slicers_t = slicers.copy()

        self.play(Create(slicer))
        self.play(ReplacementTransform(slicers_t,slicer))
        self.wait()
        self.wait(2)
        self.play(Uncreate(assi1), Uncreate(assi2), Uncreate(assi3), Uncreate(slicers))
        self.wait()

        self.play(assi.animate.move_to(ORIGIN).scale(2))
        self.wait()
        self.move_camera(phi=80* DEGREES,theta=-80 * DEGREES,run_time=2)

        surface = always_redraw(lambda: Surface(
            lambda u,v: assi.c2p(u,v,temperature(u, v)),
            u_range=[0, 4],
            v_range=[0, 2],  # dummy variable
            fill_opacity=0.1,
            stroke_color= WHITE,
            stroke_width= 0.1).set_color(WHITE))
        
        self.play(Create(surface), run_time=2)
        self.wait()
        self.play(t.animate.set_value(2), run_time=4, rate_func=linear)
        self.wait(2)

class FourierBuildUpWithLabel(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-PI, PI, PI/2],
            y_range=[-PI, PI, PI/2],
            x_length=6,
            y_length=6,
            axis_config={"include_tip": False},
        ).to_edge(LEFT)

        labels = axes.get_axis_labels(x_label="x", y_label="y")
        self.add(axes, labels)

        # Original function f(x) = x
        original = axes.plot(lambda x: x, color=MAROON_C, x_range=[-PI, PI])
        original_label = MathTex("f(x) = x").next_to(original.get_end(), UP).set_color(MAROON_C).scale(0.6)
        self.play(Create(original), Write(original_label))
        self.wait()

        # Add the general Fourier formula
        formula = MathTex(
            r"f(x) = \sum_{n=1}^{\infty} \frac{2(-1)^{n+1}}{n} \sin(nx)",
            font_size=36
        ).to_corner(UR).shift(DOWN * 0.5, LEFT * 0.5)
        
        #colors
        formula[0][25].set_color(MAROON_C)
        formula[0][0].set_color(MAROON_C)
        formula[0][1].set_color(MAROON_C)
        formula[0][2].set_color(MAROON_C)
        formula[0][3].set_color(MAROON_C)


        # General Fourier series and coefficients (MathTex block)
        fourier_block = VGroup(
            MathTex(
                r"f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left( a_n \cos(nx) + b_n \sin(nx) \right)",
                font_size=30
            ),
            MathTex(
                r"a_0 = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x)\,dx",
                font_size=28
            ),
            MathTex(
                r"a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x)\cos(nx)\,dx",
                font_size=28
            ),
            MathTex(
                r"b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x)\sin(nx)\,dx",
                font_size=28
            )
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)

        # Put it in a rectangle and position in bottom-right
        rect = SurroundingRectangle(fourier_block, color=WHITE, buff=0.4).set_color([BLUE,MAROON])
        fourier_box = VGroup(rect, fourier_block).to_corner(DR)

        # Animate appearance
        self.play(Write(fourier_box))

            
        self.play(Write(formula))           
        self.wait()

        # Initial sum function (zero)
        def zero_func(x): return 0
        cumulative = zero_func
        graph = axes.plot(cumulative, color=BLUE, x_range=[-PI, PI], stroke_width=3)
        self.add(graph)

        N_max = 6
        for n in range(1, N_max + 1):
            coef = 2 * (-1) ** (n + 1) / n
            new_term = lambda x, n=n, coef=coef: coef * np.sin(n * x)
            updated_cumulative = lambda x, cumulative=cumulative, new_term=new_term: cumulative(x) + new_term(x)

            # Plot current yellow term
            yellow_term_graph = axes.plot(new_term, x_range=[-PI, PI], color=YELLOW, stroke_opacity=0.7)
            self.play(Create(yellow_term_graph), run_time=0.8)

            # Plot new cumulative sum
            new_graph = axes.plot(updated_cumulative, color=BLUE, x_range=[-PI, PI], stroke_width=3)

            # Animate blue + yellow → new blue
            self.play(
                ReplacementTransform(graph, new_graph),
                ReplacementTransform(yellow_term_graph, new_graph),
                run_time=1.2
            )
            self.remove(graph)
            graph = new_graph
            cumulative = updated_cumulative
            self.wait(0.3)
        
        self.play(
            FadeOut(axes),
            FadeOut(labels),
            FadeOut(original),
            FadeOut(original_label),
            FadeOut(formula),
            FadeOut(graph),
            run_time=2
        )


        
        self.play(
            rect.animate.stretch_to_fit_width(config.frame_width).stretch_to_fit_height(config.frame_height).move_to(ORIGIN),
            fourier_block.animate.to_edge(UP, buff=2).shift(LEFT * 3.5).scale(1.2),
            FadeOut(fourier_block[1]),
            FadeOut(fourier_block[2]),
            FadeOut(fourier_block[3]),
            run_time=2
        )

        db_arrow = MathTex(
            r"\downarrow",
            font_size=36,
            color=WHITE
        ).next_to(fourier_block[0], DOWN, buff=0.5)
    
        complex_def = MathTex(
            r"f(x) \;=\; \sum_{n=-\infty}^{\infty} c_n \,e^{i n x}",
            font_size=36
        ).next_to(db_arrow, DOWN, buff=0.5)

        complex_coef = MathTex(
            r"c_n = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(x)\, e^{-i n x} \, dx",
            font_size=36
        ).next_to(complex_def, DOWN, buff=0.5)

        
        complex_def[0][13:25].set_color(MAROON_C)  
        complex_def[0][11:13].set_color(BLUE)
        
        complex_coef[0][0:2].set_color(BLUE) 
        
        complex = VGroup(complex_def, complex_coef)
        
        self.play(Write(complex_def),Write(db_arrow), FadeOut(rect),run_time=1)

        self.wait()
        
        self.play(Write(complex_coef), run_time=1)

        self.wait(2)
        
        merged_term = MathTex(r"c_n e^{i n x}", font_size=48).set_color(YELLOW)
        merged_term.move_to(ORIGIN)
        
        merged_term[0][0:2].set_color(BLUE)
        merged_term[0][2:6].set_color(MAROON_C)
        
        self.play(TransformMatchingShapes(complex, merged_term), Unwrite(fourier_block[0]),Uncreate(db_arrow), run_time=2)
        self.wait()

        self.play(merged_term.animate.shift(LEFT*3), run_time=2)
    
        euler_theorem = MathTex('=c_n(cos(nx)+ isin(nx))').next_to(merged_term, RIGHT, buff=0.5)
        euler_theorem[0][1:3].set_color(BLUE)
        euler_theorem[0][8:10].set_color(MAROON_C)
        euler_theorem[0][17:19].set_color(MAROON_C)

        self.play(Write(euler_theorem), run_time=2)
        self.wait(3)

        self.play(Unwrite(euler_theorem), run_time=2)
        self.wait()

        time = ValueTracker(0)

        
        vector = always_redraw(lambda: Arrow(
            start = np.array([0,0,0]),
            end = np.array([2 *np.cos(time.get_value()), 2 *np.sin(time.get_value()), 0]),
            buff = 0,
            stroke_width = 4,
            stroke_color = WHITE
        ).shift(RIGHT *2.5))

        circle = always_redraw(lambda: Circle(
            radius = 2,
            stroke_color = [MAROON_C, BLUE],
            stroke_width = 2
        ).move_to(vector.get_start()))

        module = MathTex(r'\|{C_n e^{inx}}\| = C_n').next_to(merged_term, DOWN )
        period = MathTex(r'T = \frac{2\pi}{n}').next_to(module,DOWN)

        module[0][1].set_color(BLUE)
        module[0][2].set_color(BLUE)
        module[0][3].set_color(MAROON_C)
        module[0][4].set_color(MAROON_C)
        module[0][5].set_color(MAROON_C)
        module[0][6].set_color(MAROON_C)
        module[0][9].set_color(BLUE)
        module[0][10].set_color(BLUE)

        period[0][-1].set_color(MAROON_C)


        self.play(Create(vector), Create(circle))
        self.wait()
        self.play(merged_term.animate.shift(UP*1),Write(module), Write(period))
        self.play(time.animate.increment_value(4*PI), rate_func = linear, run_time = 6)
        self.wait(2)

        gruppo1 = VGroup(vector,circle)
        vectors = VGroup()
        labels = VGroup()
        

        angle = ValueTracker(0)

        
        for i in [-2, -1, 0, 1, 2]:
            base_point = np.array([-2.5 * i, 2.5, 0])  # Spread near the top edge
            vector = always_redraw(lambda i=i, base_point=base_point: Arrow(
                start=base_point,
                end=base_point + 0.8 *np.array([
                    np.cos(angle.get_value() * -i),
                    np.sin(angle.get_value() * -i),
                    0
                ]),
                buff=0,
                stroke_width=4))
            
            vectors.add(vector)
                
            # Create a circle around the vectors
            circle = Circle(radius=1, stroke_color = [MAROON_C, BLUE],stroke_width = 2).move_to(base_point).scale(0.8)
            vectors.add(circle)

            label = MathTex(
                r"e^{ %d x i}" % (-i),
                font_size=30
            ).next_to(circle, DOWN, buff=0.2).set_color(MAROON_C)  # Label under each vector
            labels.add(label)

    
        left_vector = always_redraw(lambda: Arrow(
            start = np.array([-7, 2.5 ,0]),
            end = np.array([
                -7 + 1/2* np.cos(angle.get_value() * -3), 
                2.5 + 1/2* np.sin(angle.get_value() * -3), 
                0]),
            buff = 0
        ).set_opacity(0.3))
        
        left_circle = Circle(radius= 1/2, stroke_color = [MAROON_C, BLUE],stroke_width = 2,stroke_opacity = 0.3).move_to(np.array([-7, 2.5 ,0]))

        left_label = MathTex(
            r"e^{ %d x i}" % (-3),
            font_size=20
        ).next_to(left_circle, DOWN, buff=0.2).set_color(MAROON_C).set_opacity(0.3)  # Label under each vector

        right_vector = always_redraw(lambda: Arrow(
            start = np.array([7, 2.5 ,0]),
            end = np.array([
                7 + 1/2* np.cos(angle.get_value() * 3), 
                2.5 + 1/2* np.sin(angle.get_value() * 3), 
                0]),
            buff = 0
        ).set_opacity(0.3))
        
        right_circle = Circle(radius= 1/2, stroke_color = [MAROON_C, BLUE],stroke_width = 2, stroke_opacity = 0.3).move_to(np.array([7, 2.5 ,0]))

        right_label = MathTex(
            r"e^{ %d x i}" % (3),
            font_size=20
        ).next_to(right_circle, DOWN, buff=0.2).set_color(MAROON_C).set_opacity(0.3)

        left = VGroup(left_circle, left_vector, left_label)
        right = VGroup(right_circle, right_label, right_vector)

        self.play(ReplacementTransform(gruppo1,vectors), 
                  ReplacementTransform(merged_term,labels),
                  Unwrite(module), Unwrite(period), 
                  Create(left), Create(right), run_time = 3
                  )
        self.wait(2)
        
class Introduction(Scene):
    def construct(self):
        
        titolo = MathTex(r"Joseph \ Fourier").scale(1.5).to_edge(UP)
        self.wait()
        self.play(Write(titolo))
        self.wait(2)


        sottotitolo = MathTex(r"Theorie \ analitique \ de \ la \ chaleur").scale(0.7).to_edge(LEFT,buff = 1).shift(UP*1.8)
        sottotitolo_ita = MathTex(r"(Teoria \ analitica \  del \  calore)").scale(0.7).next_to(sottotitolo,DOWN)
        
        
        self.play(Write(sottotitolo), Write(sottotitolo_ita))

        self.wait(3)

class Conclusione(Scene):
    def construct(self):

        text = MathTex(r"X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-2\pi i \frac{kn}{N}}").scale(0.7)

        self.add(text)

class EquazioneDifferenziale(Scene):
    def construct(self):

        text = MathTex(r"\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}").scale(3)

        self.play(Write(text))
        self.wait(4)

class Newton(Scene):
    def construct(self):

        dashed_config = {
            "stroke_color": BLUE,
            "stroke_width": 3,
        }

        # Definisci la parabola come funzione paramétrica
        parabola = ParametricFunction(
            lambda t: np.array([t, -0.25 * t**2, 0]),
            t_range = [0,4],
            color = WHITE,
            **dashed_config
        )

        dashed_parabola = DashedVMobject(parabola)
        

        # Definisci l'ellisse come funzione parametrica
        ellipse = ParametricFunction(
            lambda t: np.array([3 * np.cos(t), 2*np.sin(t), 0]),
            t_range = [0, TAU],
            color = WHITE,
            **dashed_config
        )
        dashed_ellipse = DashedVMobject(ellipse, num_dashes= 25)


        t_tracker = ValueTracker(0)


        def parabola_point(t):
            return np.array([t, -0.25 * t**2, 0])

        # Funzione per posizione sull'ellisse
        def ellipse_point(t):
            return np.array([3 * np.cos(t), 2*np.sin(t), 0])

        
        image = ImageMobject("apple.png").scale(0.2)
        image2 = ImageMobject("earth.png").scale(0.2)
        image3 = ImageMobject('sun.png').scale(0.3).shift(LEFT *0.7)
        # Crea oggetto che si muove (con l'immagine)
        moving_image = always_redraw(lambda: image.move_to(parabola_point(t_tracker.get_value())))
        moving_image2 = always_redraw(lambda: image2.move_to(ellipse_point(t_tracker.get_value())))

        # Mostra la parabola
        self.play(Create(dashed_parabola), FadeIn(moving_image), run_time=2)
        self.wait(1)
        
        self.play(t_tracker.animate.set_value(4), run_time=4, rate_func= smooth)

        moving_image.clear_updaters()
        t_tracker.set_value(0)
        
        self.play(Transform(dashed_parabola, dashed_ellipse), ReplacementTransform(moving_image, moving_image2), run_time=3)
        self.play(FadeIn(image3))
        self.play(t_tracker.animate.set_value(TAU), run_time=6, rate_func=linear)
        self.wait(2)

class MobiusStripScene(ThreeDScene):
    def construct(self):
        # Parametric equation for the Möbius strip
        def mobius_surface(u, v):
            """
            Parametric equation for the Möbius strip.
            u: angle around the strip (0 to 2π)
            v: position along the strip (-1 to 1)
            """
            # Möbius strip equation
            return np.array([
                (1 + 0.5 * v * np.cos(u / 2)) * np.cos(u),
                (1 + 0.5 * v * np.cos(u / 2)) * np.sin(u),
                0.5 * v * np.sin(u / 2)
            ])

        # Create the Möbius strip surface with a gradient
        mobius = Surface(
            mobius_surface,
            u_range=[0, 2 * np.pi],
            v_range=[-1, 1],
            resolution=(60, 60),

            ).scale(3)

        # Apply a smooth color gradient across the surface
        mobius.set_color_by_gradient(MAROON_C,BLUE,MAROON_C)

        # Set the camera to a 3D perspective and rotate along Z-axis
        self.set_camera_orientation(phi=55 * DEGREES, theta=50 * DEGREES)

        
        self.play(Create(mobius))
        self.wait(1)

        self.add(mobius)
        rotation = Rotate(mobius, angle=2 * PI, axis=Z_AXIS, rate_func=linear)
        self.play(rotation, run_time=8)

        #self.wait(1)

class RectangleOnClosedCurve(Scene):
    def construct(self):
        # Define a smooth closed parametric curve (a Lissajous-like closed loop)
        def closed_curve_func(t):
            return np.array([
                np.sin(t) + 0.3 * np.sin(3 * t + 0.5) + 0.2 * np.sin(5 * t + 1.3),
                np.cos(t + 0.2) + 0.4 * np.cos(2 * t + 1.7) + 0.1 * np.cos(4 * t + 2.2),
                0
            ])

        curve = ParametricFunction(
            closed_curve_func,
            t_range=[0, TAU],
            color=BLUE,
            stroke_width=4
        )

        self.play(Create(curve))

        # Initialize 4 ValueTrackers for parameters along the curve (t in [0,1])
        alpha_trackers = [
            ValueTracker(0.0),
            ValueTracker(0.25),
            ValueTracker(0.5),
            ValueTracker(0.75),
        ]

        # Create 4 dots that will sit on the curve
        points = [Dot(radius=0.06, color=YELLOW) for _ in range(4)]

        # Attach updaters so dots follow the curve based on alpha
        for tracker, dot in zip(alpha_trackers, points):
            def updater(dot, tracker=tracker):
                alpha = tracker.get_value() % 1  # wrap around
                t = interpolate(0, TAU, alpha)
                point = closed_curve_func(t)
                dot.move_to(point)
            dot.add_updater(updater)

        # Create lines that connect the points into a quadrilateral
        lines = VGroup(
            always_redraw(lambda: Line(points[0].get_center(), points[1].get_center(), color=WHITE)),
            always_redraw(lambda: Line(points[1].get_center(), points[2].get_center(), color=WHITE)),
            always_redraw(lambda: Line(points[2].get_center(), points[3].get_center(), color=WHITE)),
            always_redraw(lambda: Line(points[3].get_center(), points[0].get_center(), color=WHITE)),
        )

        # Add everything to the scene
        self.play(*[Create(dot) for dot in points])
        self.play(*[Create(line) for line in lines])

        

        # Animate the ValueTrackers to move points smoothly around the curve
        self.wait(1/60)
        self.play(
            alpha_trackers[0].animate.increment_value(0.6),
            alpha_trackers[1].animate.increment_value(0.6),
            alpha_trackers[2].animate.increment_value(0.6),
            alpha_trackers[3].animate.increment_value(0.6),
            run_time=10
        )

        self.wait(2)

class Quote(Scene):
    def construct(self):
        quote = MathTex(
    r"\begin{gathered}"
    r" L' \ essenza \ della \ matematica \\"
    r" risiede \ proprio \ nella \ sua \ liberta' "
    r"\end{gathered}"
    ).scale(0.7).move_to(ORIGIN).shift(RIGHT*2)
        
        quotest = MathTex("-George \ Cantor").scale(0.5).next_to(quote, DOWN).shift(RIGHT*2)


        
        self.wait()
        self.play(Write(quote))
        self.play(Write(quotest))
        self.wait(4)
        self.play(Unwrite(quote), Unwrite(quotest))
        self.wait()