from manim import *
import numpy as np
from svgpathtools import svg2paths, Path
import numpy as np

def extract_points_from_svg(svg_file, n_points=2000):
    paths, _ = svg2paths(svg_file)
    # Merge all subpaths into one
    full_path = Path(*[seg for path in paths for seg in path])
    total_length = full_path.length()
    points = []
    for i in range(n_points):
        point = full_path.point(full_path.ilength(i * total_length / n_points))
        points.append(point)  # complex number
    return np.array(points)

points = extract_points_from_svg("../assets/Cat_silhouette.svg", 800)
points = np.conj(points)
points -= np.mean(points)          # Center
points /= np.max(np.abs(points))   # Normalize
points *= 3.0
np.save("drawing_points.npy", points)

class FourierEpicyclesCat(Scene):
    def construct(self):
        # Load points (complex numbers)
        path_points = np.load("drawing_points.npy")
        n = len(path_points)

        # Compute Fourier coefficients
        fourier = []
        for k in range(-n // 2, n // 2):
            coef = np.sum(path_points * np.exp(-2j * np.pi * k * np.arange(n) / n)) / n
            fourier.append((k, coef))
        fourier.sort(key=lambda x: abs(x[1]), reverse=True)

        # Time tracker
        time = ValueTracker(0)
        self.add(time)

        # Draw rotating vectors and circles
        def get_vectors():
            origin = 0 + 0j
            arrows = []
            for freq, coef in fourier[:300]:  # top N terms
                angle = 2 * np.pi * freq * time.get_value()
                tip = origin + coef * np.exp(1j * angle)
                
                # Create an arrow (vector)
                arrow = Arrow(
                    start=complex_to_3d(origin),
                    end=complex_to_3d(tip),
                    buff=0,
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.05,
                )
                arrows.append(arrow)
                origin = tip
            return VGroup(*arrows)

        def get_circles():
            origin = 0 + 0j
            circles = []
            for freq, coef in fourier[:100]:  # top N terms
                # Create a circle surrounding the vector
                circle = Circle(
                    radius=abs(coef),  # The radius is the magnitude of the Fourier coefficient
                    color=[MAROON_C, BLUE],
                    stroke_width=1,
                    stroke_opacity = 0.3,
                ).move_to(complex_to_3d(origin))  # Position the circle at the origin of the vector
                circles.append(circle)
                origin += coef * np.exp(1j * 2 * np.pi * freq * time.get_value())
            return VGroup(*circles)

        # Create the VGroups for the vectors and circles
        vector_group = always_redraw(get_vectors)
        circle_group = always_redraw(get_circles)

        # Add the groups to the scene
        self.add(vector_group, circle_group)

        # Trace the final tip
        def get_tip():
            origin = 0 + 0j
            for freq, coef in fourier[:100]:
                angle = 2 * np.pi * freq * time.get_value()
                origin += coef * np.exp(1j * angle)
            return complex_to_3d(origin)

        trace = TracedPath(get_tip, stroke_color=BLUE, stroke_width=2)
        self.add(trace)

        # Animate!
        self.play(time.animate.set_value(1.1), run_time=20, rate_func=linear)
        

def complex_to_3d(z):
    return np.array([z.real, z.imag, 0.0])




