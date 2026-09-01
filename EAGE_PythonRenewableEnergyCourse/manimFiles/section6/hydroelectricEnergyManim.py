"""
Hydroelectric Potential Energy Animation: E = m·g·h
=====================================================
Run with:
    manim -pql hydroelectricEnergyManim.py EnergyAnimation

For high quality:
    manim -pqh hydroelectricEnergyManim.py EnergyAnimation

Requires: pip install manim
"""

from manim import *

# ── Colour palette ──────────────────────────────────────────────────────────
C_WATER   = "#4FC3F7"
C_ENERGY  = "#FFD54F"
C_MASS    = "#81C784"
C_GRAVITY = "#FF8A65"
C_HEIGHT  = "#CE93D8"
C_BG      = "#0D1B2A"
C_TEXT    = WHITE
C_DIM     = "#607D8B"


# ── Helper: coloured equation term ──────────────────────────────────────────
def coloured_eq():
    eq = MathTex(
        r"E", r"=", r"m", r"\cdot", r"g", r"\cdot", r"h",
        font_size=72,
    )
    eq[0].set_color(C_ENERGY)
    eq[2].set_color(C_MASS)
    eq[4].set_color(C_GRAVITY)
    eq[6].set_color(C_HEIGHT)
    return eq


# ════════════════════════════════════════════════════════════════════════════
class EnergyAnimation(Scene):
    """All scenes combined into one cohesive animation."""

    def construct(self):
        self.camera.background_color = C_BG
        self._scene_title()
        self._scene_diagram()
        self._scene_vary_h()
        self._scene_vary_m()
        self._scene_vary_g()
        self._scene_summary()

    # ── 1  Title card ────────────────────────────────────────────────────────
    def _scene_title(self):
        title = Text("Hydroelectric Potential Energy", font_size=48, color=C_TEXT)
        subtitle = Text("E = m · g · h", font_size=36, color=C_ENERGY)
        subtitle.next_to(title, DOWN, buff=0.4)

        tag_m = Text("m = mass of water", font_size=22, color=C_MASS)
        tag_g = Text("g = gravitational acceleration", font_size=22, color=C_GRAVITY)
        tag_h = Text("h = hydraulic head (height)", font_size=22, color=C_HEIGHT)
        tag_E = Text("E = potential energy", font_size=22, color=C_ENERGY)

        tags = VGroup(tag_E, tag_m, tag_g, tag_h).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        tags.next_to(subtitle, DOWN, buff=0.55)

        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.play(LaggedStart(*[FadeIn(t, shift=RIGHT * 0.2) for t in tags], lag_ratio=0.25))
        self.wait(1.5)
        self.play(FadeOut(VGroup(title, subtitle, tags)))

    # ── 2  Diagram: reservoir → turbine ─────────────────────────────────────
    def _scene_diagram(self):
        mountain = Polygon(
            [-5.5, -2.5, 0], [-3.0, 1.8, 0], [-0.5, -2.5, 0],
            color=GRAY_D, fill_color=GRAY_D, fill_opacity=1,
        )

        reservoir = Rectangle(width=2.2, height=0.9, color=C_WATER,
                               fill_color=C_WATER, fill_opacity=0.85)
        reservoir.move_to([-3.0, 1.15, 0])
        res_label = Text("Reservoir", font_size=18, color=C_TEXT)
        res_label.next_to(reservoir, UP, buff=0.12)

        penstock = Line([-2.2, 0.7, 0], [0.5, -1.8, 0],
                        color=GRAY_B, stroke_width=10)

        turbine = Circle(radius=0.28, color=YELLOW, fill_color=YELLOW_E,
                         fill_opacity=1).move_to([0.7, -2.0, 0])
        turbine_label = Text("Turbine", font_size=16, color=C_TEXT)
        turbine_label.next_to(turbine, DOWN, buff=0.08)

        h_start = [-0.4, -2.0, 0]
        h_end   = [-0.4,  0.7, 0]
        h_arrow = DoubleArrow(h_start, h_end, color=C_HEIGHT,
                              buff=0, stroke_width=3, tip_length=0.18)
        h_label = MathTex(r"h", color=C_HEIGHT, font_size=42)
        h_label.next_to(h_arrow, RIGHT, buff=0.1)

        drop = Dot(radius=0.18, color=C_WATER).move_to([-2.2, 0.7, 0])

        eq = coloured_eq()
        eq.move_to([3.2, 0.5, 0])
        eq_box = SurroundingRectangle(eq, color=C_ENERGY, buff=0.2,
                                      corner_radius=0.15, stroke_width=2)
        caption = Text("Falling water converts\npotential energy → electricity",
                       font_size=20, color=C_DIM, line_spacing=1.1)
        caption.next_to(eq_box, DOWN, buff=0.4)

        self.play(
            FadeIn(mountain), FadeIn(reservoir), FadeIn(res_label),
            Create(penstock), run_time=1.0,
        )
        self.play(FadeIn(turbine), FadeIn(turbine_label))
        self.play(GrowArrow(h_arrow), Write(h_label))
        self.play(Write(eq), Create(eq_box))
        self.play(FadeIn(caption))

        self.play(
            MoveAlongPath(drop, penstock),
            Rotate(turbine, angle=2 * PI, about_point=turbine.get_center()),
            run_time=2.0, rate_func=linear,
        )
        self.wait(1)

        self.play(FadeOut(VGroup(
            mountain, reservoir, res_label, penstock,
            turbine, turbine_label, h_arrow, h_label,
            drop, eq, eq_box, caption,
        )))

    # ── 3  Vary h (height) ──────────────────────────────────────────────────
    def _scene_vary_h(self):
        self._parameter_scene(
            param_name="h",
            param_color=C_HEIGHT,
            param_label="Hydraulic Head  h  (height of water above turbine)",
            values=[("h = 20 m\n(low dam)",    1000, 9.81, 20),
                    ("h = 60 m\n(medium dam)", 1000, 9.81, 60),
                    ("h = 100 m\n(high dam)",  1000, 9.81, 100)],
            fixed_note="m = 1000 kg   ·   g = 9.81 m/s²",
            interpretation=(
                "Double the height → double the energy.\n"
                "Taller dams are significantly more powerful."
            ),
        )

    # ── 4  Vary m (mass) ─────────────────────────────────────────────────────
    def _scene_vary_m(self):
        self._parameter_scene(
            param_name="m",
            param_color=C_MASS,
            param_label="Mass of water  m  (flow rate × time)",
            values=[("m = 500 kg\n(trickle)",      500,  9.81, 60),
                    ("m = 1000 kg\n(moderate flow)", 1000, 9.81, 60),
                    ("m = 2000 kg\n(large flow)",    2000, 9.81, 60)],
            fixed_note="g = 9.81 m/s²   ·   h = 60 m",
            interpretation=(
                "More water volume per second → more energy.\n"
                "High-flow rivers compensate for lower heads."
            ),
        )

    # ── 5  Vary g (gravity) ──────────────────────────────────────────────────
    def _scene_vary_g(self):
        self._parameter_scene(
            param_name="g",
            param_color=C_GRAVITY,
            param_label="Gravitational acceleration  g",
            values=[("Moon\ng ≈ 1.62 m/s²",    1000, 1.62, 60),
                    ("Earth\ng = 9.81 m/s²",   1000, 9.81, 60),
                    ("Jupiter\ng ≈ 24.8 m/s²", 1000, 24.8, 60)],
            fixed_note="m = 1000 kg   ·   h = 60 m",
            interpretation=(
                "g is constant on Earth (9.81 m/s²).\n"
                "Higher gravity → more force on the same water mass."
            ),
        )

    # ── Generic parameter-variation scene ────────────────────────────────────
    def _parameter_scene(self, param_name, param_color,
                         param_label, values, fixed_note, interpretation):
        title = Text(f"Varying  {param_name}", font_size=38,
                     color=param_color, weight=BOLD)
        title.to_edge(UP, buff=0.35)

        param_text = Text(param_label, font_size=22, color=C_TEXT)
        param_text.next_to(title, DOWN, buff=0.2)

        fixed = Text(fixed_note, font_size=19, color=C_DIM)
        fixed.next_to(param_text, DOWN, buff=0.15)

        self.play(Write(title), FadeIn(param_text), FadeIn(fixed))

        eq = coloured_eq()
        eq.next_to(fixed, DOWN, buff=0.4)
        self.play(Write(eq))

        max_E = max(m * g * h for _, m, g, h in values)
        bar_group  = VGroup()
        bar_labels = VGroup()
        E_labels   = VGroup()

        bar_area_width = 6.0
        bar_area_left  = -bar_area_width / 2
        bar_max_height = 2.0
        bar_width      = bar_area_width / len(values) * 0.55
        spacing        = bar_area_width / len(values)

        for i, (case_label, m_val, g_val, h_val) in enumerate(values):
            E = m_val * g_val * h_val
            bar_h = (E / max_E) * bar_max_height
            x = bar_area_left + spacing * (i + 0.5)
            y_bottom = -2.8

            bar = Rectangle(
                width=bar_width, height=bar_h,
                color=param_color, fill_color=param_color, fill_opacity=0.85,
            )
            bar.move_to([x, y_bottom + bar_h / 2, 0])

            case_txt = Text(case_label, font_size=16, color=C_TEXT, line_spacing=1.0)
            case_txt.move_to([x, y_bottom - 0.45, 0])

            E_val_txt = Text(f"E ≈ {E:,.0f} J", font_size=17, color=C_ENERGY)
            E_val_txt.next_to(bar, UP, buff=0.12)

            bar_group.add(bar)
            bar_labels.add(case_txt)
            E_labels.add(E_val_txt)

        baseline = Line([-3.5, -2.8, 0], [3.5, -2.8, 0], color=C_DIM, stroke_width=1.5)
        self.play(Create(baseline))

        for bar, lbl, elbl in zip(bar_group, bar_labels, E_labels):
            self.play(
                GrowFromEdge(bar, DOWN),
                FadeIn(lbl, shift=UP * 0.15),
                FadeIn(elbl),
                run_time=0.7,
            )

        interp = Text(interpretation, font_size=20, color=C_TEXT, line_spacing=1.2)
        interp.to_edge(RIGHT, buff=0.5)
        interp.shift(DOWN * 0.8)
        bubble = SurroundingRectangle(interp, color=param_color, buff=0.2,
                                      corner_radius=0.12, stroke_width=1.5)
        self.play(FadeIn(bubble), Write(interp))
        self.wait(2.5)

        self.play(FadeOut(VGroup(
            title, param_text, fixed, eq,
            bar_group, bar_labels, E_labels,
            baseline, interp, bubble,
        )))

    # ── 6  Summary ──────────────────────────────────────────────────────────
    def _scene_summary(self):
        title = Text("Summary", font_size=44, color=C_TEXT, weight=BOLD)
        title.to_edge(UP, buff=0.45)

        eq = coloured_eq()
        eq.next_to(title, DOWN, buff=0.5)

        rules = VGroup(
            Text("↑ h  →  ↑ E   (taller dam, more energy)",
                 font_size=24, color=C_HEIGHT),
            Text("↑ m  →  ↑ E   (more water volume, more energy)",
                 font_size=24, color=C_MASS),
            Text("g = 9.81 m/s²  fixed on Earth",
                 font_size=24, color=C_GRAVITY),
            Text("E scales linearly with each factor",
                 font_size=22, color=C_DIM),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        rules.next_to(eq, DOWN, buff=0.55)

        footer = Text(
            "Hydroelectric power = converting stored gravitational\n"
            "potential energy of water into electrical energy.",
            font_size=20, color=C_DIM, line_spacing=1.2,
        )
        footer.to_edge(DOWN, buff=0.4)

        self.play(Write(title))
        self.play(Write(eq))
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rules],
                               lag_ratio=0.3))
        self.play(FadeIn(footer))
        self.wait(3)
        self.play(FadeOut(VGroup(title, eq, rules, footer)))

        final = MathTex(r"E = m \cdot g \cdot h", font_size=96, color=C_ENERGY)
        self.play(Write(final))
        self.play(final.animate.scale(1.15), rate_func=there_and_back, run_time=1.2)
        self.wait(1.5)
        self.play(FadeOut(final))
