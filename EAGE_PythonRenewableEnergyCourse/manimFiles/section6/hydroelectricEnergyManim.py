from manim import *


class HydroelectricEnergy(Scene):
    def construct(self):
        # ── 1. Display the Formula ────────────────────────────────────────────
        formula = MathTex("E", "=", "m", "\\cdot", "g", "\\cdot", "h")
        formula.set_color_by_tex("E", YELLOW)
        formula.set_color_by_tex("m", BLUE)
        formula.set_color_by_tex("h", GREEN)
        formula.to_edge(UP)

        self.play(Write(formula))
        self.wait(1)

        # ── 2. Setup visual elements ──────────────────────────────────────────
        # Ground line + turbine circle
        ground = Line(LEFT * 3, RIGHT * 3).shift(DOWN * 2.5)
        turbine = Circle(radius=0.3, color=GRAY).move_to(ground.get_center())
        turbine_label = Text("Turbine", font_size=20).next_to(turbine, DOWN)

        # Water box – represents mass m
        water_box = Rectangle(
            width=1, height=1, fill_opacity=0.8, color=BLUE
        )
        water_box.move_to(UP * 1)  # Initial height h ≈ 1

        # Height arrow from ground to bottom of water box
        height_arrow = DoubleArrow(
            start=ground.get_center(),
            end=water_box.get_bottom(),
            buff=0,
            color=GREEN,
        )
        h_label = MathTex("h", color=GREEN).next_to(height_arrow, LEFT)
        m_label = MathTex("m", color=BLUE).move_to(water_box.get_center())

        self.play(Create(ground), Create(turbine), Write(turbine_label))
        self.play(
            FadeIn(water_box),
            Create(height_arrow),
            Write(h_label),
            Write(m_label),
        )
        self.wait(1)

        # ── 3. Vary Height (h) ────────────────────────────────────────────────
        # Move water higher → h increases → E increases
        self.play(
            water_box.animate.shift(UP * 1.5),
            height_arrow.animate.stretch_to_fit_height(
                4, about_edge=DOWN
            ).shift(UP * 0.75),
            h_label.animate.shift(UP * 0.75),
            run_time=2,
        )
        # Flash the E and h terms in the formula
        self.play(
            Indicate(formula[0]),   # E
            Indicate(formula[6]),   # h
        )
        self.wait(1)

        # ── 4. Vary Mass (m) ──────────────────────────────────────────────────
        # Widen the water box → m doubles → E doubles
        new_m_label = MathTex("2m", color=BLUE).move_to(
            water_box.get_center() + RIGHT * 0.5
        )
        self.play(
            water_box.animate.stretch_to_fit_width(2, about_edge=LEFT),
            Transform(m_label, new_m_label),
            run_time=2,
        )
        self.play(
            Indicate(formula[0]),   # E
            Indicate(formula[2]),   # m
        )
        self.wait(1)

        # ── 5. Release – water falls to turbine ───────────────────────────────
        self.play(
            water_box.animate.move_to(turbine.get_center() + UP * 0.3),
            FadeOut(height_arrow),
            FadeOut(h_label),
            FadeOut(m_label),
            run_time=1.5,
            rate_func=rush_into,
        )
        self.play(Rotate(turbine, angle=2 * PI * 3, run_time=1))
        turbine.set_color(YELLOW)

        # ── 6. Final message ──────────────────────────────────────────────────
        final_text = Text(
            "Higher h  +  More m  =  More Power",
            font_size=28,
            color=YELLOW,
        )
        final_text.next_to(formula, DOWN, buff=0.4)
        self.play(Write(final_text))
        self.wait(2)
