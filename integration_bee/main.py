from manim import *
from math import *


config.pixel_width = 1080
config.pixel_height = 1920

config.frame_width = 9
config.frame_height = 16


class Scena1(Scene):
    def construct(self):
        axes = Axes(x_range = [-1,5,1], 
                    y_range= [-1,4,1], 
                    x_length= 7,
                    y_length= 6).scale(0.5).to_edge(UP, buff= 4)

        
        axes.get_x_axis().add_numbers(font_size = 24, excluding = [-1])

        graph_min_left = axes.plot(
            lambda x: x**4,
            x_range=[0,1],
            color = TEAL_E
        )
        
        graph_min_right = axes.plot(
            lambda x: 1/(x**4),
            x_range=[1,4],
            color = MAROON_C
        )
        
        graph_cut_left = axes.plot(
            lambda x: 1/(x**4),
            x_range=[0.715,1],
            color = MAROON_C
        )
        
        graph_cut_right = axes.plot(
            lambda x: x**4,
            x_range=[1,1.4],
            color = TEAL_E
        )

        area_left = axes.get_area(
            graph_min_left,
            x_range=[0,1],
            color=TEAL_E,
            opacity=0.5
        )
        
        area_right = axes.get_area(
            graph_min_right,
            x_range=[1,4],
            color=MAROON_C,
            opacity=0.5
        )

        graph1 = axes.plot(lambda x: x**4, x_range = [0,1.4])
        graph2 = axes.plot(lambda x: 1/x**4, x_range = [0.715,4])

        simbolo = MathTex(r"=").next_to(axes,DOWN,buff = 1).scale(0.7)
        titolo = MathTex(r"\int_0^4 \text{min}\Bigg(\frac{1}{x^4}, x^4\Bigg)dx").scale(0.7)
        
        titolo[0][7:11].set_color(MAROON_C)
        titolo[0][12:14].set_color(TEAL_D)
        
        riga_titolo = VGroup(simbolo, titolo)
        riga_titolo.arrange(RIGHT, buff=0.2).shift(DOWN*0.5)


        integranda = MathTex(r"\text{min}\Bigg(\frac{1}{x^4}, x^4\Bigg)").scale(0.7).next_to(axes,UP,buff= 0.5)


        passo1 = MathTex(r"= \int_0^1 x^4 dx + \int_1^4 \frac{1}{x^4}dx").scale(0.7).next_to(titolo,DOWN,buff= 0.5)
        passo1.align_to(riga_titolo)

        passo1[0][4:6].set_color(TEAL_E)
        passo1[0][12:16].set_color(MAROON_C)


        passo2 = MathTex(r"\frac{x^5}{5}\Bigg]^1_0 - \frac{1}{3x^3}\Bigg]^4_1").scale(0.7).next_to(passo1,DOWN,buff= 0.5)
        passo2.align_to(passo1)

        passi = VGroup(passo1,passo2)

        passo_finale = MathTex(r"= \frac{1}{5} + \frac{63}{192}").scale(0.7).next_to(riga_titolo, DOWN, buff= 0.5).align_to(riga_titolo)
        res= MathTex(r"= \frac{169}{320}").scale(0.7).next_to(passo_finale,RIGHT,buff= -0.3)
        
        
        self.play(Create(axes))
        
        self.play(Create(graph1))
        self.play(Create(graph2))

        self.wait()

        self.play(Write(integranda))
        
        self.wait()

        self.play(
            FadeIn(graph_cut_left),
            FadeIn(graph_cut_right),
            FadeIn(graph_min_left),
            FadeIn(graph_min_right),
            integranda[0][4:8].animate.set_color(MAROON_C),
            integranda[0][9:11].animate.set_color(TEAL_E)
            )


        self.remove(graph1,graph2)

        self.wait()

        self.play(graph_cut_right.animate.set_opacity(0.2), graph_cut_left.animate.set_opacity(0.2))

        self.play(FadeIn(area_left), FadeIn(area_right))

        self.play(ReplacementTransform(integranda, riga_titolo), run_time = 1)
        self.wait()
        self.play(Write(passo1))
        self.wait()
        self.play(Write(passo2))
        self.wait()
        self.play(ReplacementTransform(passi, passo_finale))
        self.play(passo_finale.animate.shift(LEFT*0.5))
        self.play(Write(res))
        self.wait()
        self.play(Indicate(res[0][-7:]))
        self.wait(1)


class Scena2(Scene):
    def construct(self):
        
        # Grafico 1
        
        axes = Axes(x_range = [-1,7/2 * PI ,1], 
                    y_range= [-1,2,1], 
                    x_length= 10,
                    y_length= 6)
        
        x_labels = {
            PI/2: MathTex(r"\frac{\pi}{2}"),
            PI: MathTex(r"\pi"),
            3*PI/2: MathTex(r"\frac{3\pi}{2}"),
            2*PI: MathTex(r"2\pi"),
            5*PI/2: MathTex(r"\frac{5\pi}{2}"),
            3*PI: MathTex(r"3\pi"),
        }

        axes.x_axis.add_labels(x_labels)

        axes.scale(0.5).to_edge(UP,buff= 3)
        graph = axes.plot(lambda x: cos(x)**2)
        area = axes.get_area(graph= graph, x_range = [0, 3*PI])

        #Formule
            
        integrale = MathTex(r"\int_0^{3 \pi}\text{cos}^2(x)dx").scale(0.7).next_to(axes,DOWN,buff = 0.5)
        uguale = MathTex(r"=").scale(0.7).next_to(integrale, RIGHT,buff= 0.2)
        
        bisezione1 = MathTex(r"\text{cos}^2(x)  = ").scale(0.7).shift(DOWN*2, LEFT *0.5)
        bisezione2= MathTex(r"\frac{1}{2}\big(1+ \text{cos}(2x)\big)").scale(0.7).next_to(bisezione1,RIGHT,buff= 0.2)

        bisezione = VGroup(bisezione1,bisezione2).to_edge(DOWN, buff=3).shift(LEFT*0.4)
        box = SurroundingRectangle(bisezione, buff= 0.2).set_color_by_gradient([TEAL_C,BLUE_D])


        int2 = MathTex(r"\int_0^{3 \pi} \frac{1}{2}\big(1+ \text{cos}(2x)\big)dx =").scale(0.7).next_to(integrale,DOWN,buff= 0.5)
        int3 = MathTex(r" \frac{1}{2}\Big[x + \frac{1}{2}\text{sin}(2x)\Big]^{3 \pi}_0 =").scale(0.7).next_to(int2,DOWN,buff=0.5)
        sol = MathTex(r'\frac{3\pi}{2}').scale(0.7).next_to(int3,RIGHT,buff=0.2)


        #Animazioni
        
        self.play(Write(axes))
        self.play(Create(graph))
        self.wait()
        self.play(Write(integrale))
        self.play(FadeIn(area))
        self.wait()
        self.play(Write(uguale))
        self.play(Write(bisezione), Create(box))
        self.wait()
        self.play(Indicate(bisezione1, color = [TEAL_C,BLUE_D]), Indicate(integrale[0][-9:-3],color = [TEAL_C,BLUE_D]))
        self.wait()
        self.play(ReplacementTransform(VGroup(integrale.copy(), bisezione2.copy()), int2))
        self.wait()
        self.play(Unwrite(bisezione), Uncreate(box), run_time = 0.7)
        self.wait()
        self.play(Write(int3))
        self.wait()
        self.play(Write(sol))
        self.wait(2)


class Scena3(Scene):
    def construct(self):

        # Assi e grafico 

        axes = Axes(x_range = [-1,5/2 * PI ,1], 
                    y_range= [-1,2,1], 
                    x_length= 10,
                    y_length= 6)
        

        x_labels = {
            PI/2: MathTex(r"\frac{\pi}{2}"),
            PI: MathTex(r"\pi"),
            3*PI/2: MathTex(r"\frac{3\pi}{2}"),
            2*PI: MathTex(r"2\pi"),
        }

        axes.x_axis.add_labels(x_labels)
        axes.scale(0.4).to_edge(UP,buff= 3)

        graph = axes.plot(lambda x: sin(log(x)), x_range =[0.01,5/2*PI])
        area = axes.get_area(graph, x_range = [PI,2*PI])
        

        # Formule

        integrale = MathTex(r"\int_{\pi}^{2\pi}\text{sin}(\text{ln}(x))dx").scale(0.7).next_to(axes,DOWN,buff = 0.7)
        uguale = MathTex(r'=').scale(0.7).next_to(integrale,buff =0.2)

        s1 = MathTex(r"t = \text{ln}(x)").scale(0.7).next_to(integrale,DOWN,buff= 1)
        s2 = MathTex(r"dt=  \frac{1}{x}dx").scale(0.7).next_to(s1,DOWN,buff=0.4)
        s3 = MathTex(r"x = e^t").scale(0.7).next_to(s2,DOWN,buff=0.4)

    
        s = VGroup(s1,s2,s3)
        box = SurroundingRectangle(s, color = [TEAL_E,BLUE_D], buff = 0.2)


        #Animazion1
        self.add(axes, graph,area, integrale, s,box, uguale)
        self.play(VGroup(s,box).animate.shift(LEFT*2))
        self.wait()

        #Estremi
        
        estremo1 = MathTex(r"x = \pi \implies t =\text{ln}(\pi)").scale(0.7).next_to(s1,RIGHT,buff = 0.7)
        estremo2 = MathTex(r"x = 2\pi \implies t =\text{ln}(2\pi)").scale(0.7).next_to(estremo1,DOWN,buff= 0.4, aligned_edge=LEFT)
        dx = MathTex(r"dx= xdt \implies dx=e^tdt").scale(0.7).next_to(estremo2,DOWN,buff = 0.4, aligned_edge=LEFT)
        

        self.play(ReplacementTransform(integrale[0][4].copy(), estremo1[0][:3]))
        self.play(Write(estremo1[0][3:]))
        
        self.wait()

        self.play(ReplacementTransform(integrale[0][1:3].copy(), estremo2[0][:4]))
        self.play(Write(estremo2[0][4:]))

        self.wait()
        self.play(Write(dx))
        self.wait()

        #Nuovo int
        int2 = MathTex(r"\int_{\text{ln}(\pi)}^{\text{ln}(2\pi)}e^t \sin(t)dt").scale(0.7).next_to(integrale, DOWN,buff = 4)
        

        self.play(Write(int2[0][0]))
        self.play(ReplacementTransform(estremo1[0][-7:].copy(), int2[0][7:12]))
        self.wait()
        self.play(ReplacementTransform(estremo2[0][-8:].copy(), int2[0][1:7]))
        self.play(Write(int2[0][12:]))
        self.wait(2)

        self.play(Uncreate(box), Unwrite(s), Unwrite(estremo1), Unwrite(estremo2), Unwrite(dx))
        
        self.play(int2.animate.next_to(integrale,DOWN,buff=0.4))

        uguale2 = MathTex(r"=").scale(0.7).next_to(int2,RIGHT,buff = 0.2)
        self.play(Write(uguale2))
        
        self.wait()

        # Integrazione per parti1

        parti = MathTex(r"\int f g' = fg-\int f'g").scale(0.7).next_to(int2,DOWN,buff=0.7)
        box2= SurroundingRectangle(parti, color = [TEAL_E,BLUE_D],buff=0.2)
        
        f1 = MathTex(r"f = \sin(t) \implies f' = \cos(t)").scale(0.7).next_to(box2,DOWN,buff=0.8)
        g1 = MathTex(r"g' = e^t \implies g = e^t").scale(0.7).next_to(f1,DOWN,buff=0.4)

        parti_nuovo = MathTex(r"\int_{\text{ln}(\pi)}^{\text{ln}(2\pi)}\sin(t)e^tdt = \sin(t) e^t - \int_{\text{ln}(\pi)}^{\text{ln}(2\pi)}\cos(t)e^tdt").scale(0.5).next_to(box2,DOWN,buff=0.8)
        
        #colori

        parti[0][1].set_color(TEAL_D)
        parti[0][5].set_color(TEAL_D)
        parti[0][9:11].set_color(TEAL_D)


        parti[0][2:4].set_color(MAROON_C)
        parti[0][6].set_color(MAROON_C)
        parti[0][-1].set_color(MAROON_C)



        f1[0][0].set_color(TEAL_D)
        f1[0][10:12].set_color(TEAL_D)
        
        g1[0][:2].set_color(MAROON_C)
        g1[0][7].set_color(MAROON_C)


        parti_nuovo[0][12:18].set_color(TEAL_D)
        parti_nuovo[0][18:20].set_color(MAROON_C)
        parti_nuovo[0][23:29].set_color(TEAL_D)
        parti_nuovo[0][29:31].set_color(MAROON_C)
        parti_nuovo[0][-10:-4].set_color(TEAL_D)
        parti_nuovo[0][-4:-2].set_color(MAROON_C)
        

        #Animazioni
        self.play(Write(parti), Create(box2))

        self.play(Write(f1),Write(g1))
        self.wait()
        self.play(VGroup(f1,g1).animate.shift(DOWN*2))
        self.wait()
        self.play(ReplacementTransform(int2.copy(),parti_nuovo[0][:22]))
        self.play(Write(parti_nuovo[0][22]))
        
        self.play(ReplacementTransform(f1[0][3:9].copy(),parti_nuovo[0][23:29]))
        self.play(ReplacementTransform(g1[0][-2:].copy(),parti_nuovo[0][29:31]))
        
        self.play(Write(parti_nuovo[0][31:-10]))

        self.play(ReplacementTransform(f1[0][-6:].copy(),parti_nuovo[0][-10:-4]))
        self.play(ReplacementTransform(g1[0][3:5].copy(),parti_nuovo[0][-4:-2]))

        self.play(Write(parti_nuovo[0][-2:]))
        self.wait()


        #Parti2 

        f2= MathTex(r"f= \cos(t) \implies f' = -\sin(t)").scale(0.7).move_to(f1)
        g2 = MathTex(r"g'=e^t \implies g = e^t").scale(0.7).move_to(g1)

        parti_nuovo2= MathTex(r'\int_{\text{ln}(\pi)}^{\text{ln}(2\pi)}e^t\sin(t)dt = \sin(t) e^t - \Big(\cos(t)e^t -\int_{\text{ln}(\pi)}^{\text{ln}(2\pi)}-\sin(t)e^tdt \Big)').scale(0.5).move_to(parti_nuovo)
        
        #Colori
        f2[0][0].set_color(TEAL_D)
        f2[0][10:12].set_color(TEAL_D)

        g2[0][:2].set_color(MAROON_C)
        g2[0][7].set_color(MAROON_C)
                
            
        parti_nuovo2[0][33:39].set_color(TEAL_D)
        parti_nuovo2[0][-12:-5].set_color(TEAL_D)

        parti_nuovo2[0][39:41].set_color(MAROON_C)
        parti_nuovo2[0][-5:-3].set_color(MAROON_C)


        self.play(parti_nuovo.animate.set_color(WHITE))

        
        self.play(parti_nuovo[0][-10:-4].animate.set_color(TEAL_D),
                    parti_nuovo[0][-4:-2].animate.set_color(MAROON_C))
        
    

        self.play(ReplacementTransform(f1,f2))
        self.play(ReplacementTransform(g1,g2))

        self.play(parti_nuovo.animate.shift(LEFT*0.82))
        
        self.play(ReplacementTransform(VGroup(parti_nuovo[0][32:],f2[0][:8].copy()),parti_nuovo2[0][32:39]))
        self.play(ReplacementTransform(g2[0][-4:].copy(),parti_nuovo2[0][39:41]))

        self.play(Write(parti_nuovo2[0][41:54]))

        self.play(ReplacementTransform(f2[0][-10:].copy(),parti_nuovo2[0][54:61]))
        self.play(ReplacementTransform(g2[0][-4:].copy(),parti_nuovo2[0][61:63]))
        self.play(Write(parti_nuovo2[0][-3:]))

        tutto = VGroup(parti_nuovo, parti_nuovo2)
        finale = MathTex(r"\int_{\text{ln}(\pi)}^{\text{ln}(2\pi)}e^t\sin(t)dt = \frac{\sin(t) e^t - \cos(t)e^t}{2}\Bigg ]^{\ln(2\pi)}_{\ln(\pi)}").scale(0.5).move_to(tutto)
        res = MathTex(r"=\pi \left( \sin(\ln(2\pi)) - \cos(\ln(2\pi)) \right) - \frac{\pi}{2} \left( \sin(\ln(\pi)) - \cos(\ln(\pi)) \right)").scale(0.5).next_to(finale,DOWN,buff = 0.4)

        
        # 1. Rimuovi gli elementi di supporto
        self.play(Unwrite(f2), Unwrite(g2), Unwrite(parti), Uncreate(box2))

        # 2. Definiamo le nuove righe di semplificazione
        no_parentesi = MathTex(r"\int_{\ln(\pi)}^{\ln(2\pi)}e^t\sin(t)dt = \sin(t) e^t - \cos(t)e^t -\int_{\ln(\pi)}^{\ln(2\pi)}\sin(t)e^tdt").scale(0.5).next_to(int2,DOWN,buff=0.4)
        no_parentesi2 = MathTex(r"\implies 2\int_{\ln(\pi)}^{\ln(2\pi)}e^t\sin(t)dt = \sin(t) e^t - \cos(t)e^t ").scale(0.5).next_to(no_parentesi, DOWN, buff=0.4)
        no_parentesi3 = MathTex(r"\implies \int_{\ln(\pi)}^{\ln(2\pi)}e^t\sin(t)dt = \frac{\sin(t) e^t - \cos(t)e^t}{2}").scale(0.5).next_to(no_parentesi2, DOWN, buff=0.4)

        # 3. Trasformazione: 'tutto' scompare e diventa 'no_parentesi'
        self.play(ReplacementTransform(tutto, no_parentesi))
        self.wait()
        
        # 4. Sviluppo algebrico (NON usare 'tutto.animate' qui, è già sparito!)
        self.play(Write(no_parentesi2))
        self.wait()
        self.play(Write(no_parentesi3))
        self.wait()

        finale = MathTex(
            r"\int_{\ln(\pi)}^{\ln(2\pi)}e^t\sin(t)dt = \left[ \frac{e^t(\sin(t) - \cos(t))}{2} \right]_{\ln(\pi)}^{\ln(2\pi)}"
        ).scale(0.5).next_to(int2,DOWN,buff=0.4)
        
        res = MathTex(
            r"= \pi (\sin(\ln(2\pi)) - \cos(\ln(2\pi))) - \frac{\pi}{2} (\sin(\ln(\pi)) - \cos(\ln(\pi)))"
        ).scale(0.5).next_to(finale, DOWN, buff=0.4)

        # Usiamo un VGroup temporaneo per la trasformazione finale
        vecchie_righe = VGroup(no_parentesi, no_parentesi2, no_parentesi3)
        
        # Le righe di calcolo (vecchie_righe) salgono e si trasformano in 'finale'
        # Contemporaneamente, int2 e uguale2 svaniscono per fargli spazio
        self.play(
            ReplacementTransform(vecchie_righe, finale))
        self.wait()
        
        self.play(Write(res))
        self.wait(2)