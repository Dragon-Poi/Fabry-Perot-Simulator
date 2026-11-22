"""Fabry–Perot 干涉仪教学动画。依赖 NumPy 和 Matplotlib。"""

import math
from dataclasses import dataclass
from weakref import WeakSet

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Button, Slider, TextBox


# 长度和时间分别以镜距 e 和单程时间 e/c 为单位。
MIRROR_SPACING = 1.0
C = 1.0
SOURCE_X = -MIRROR_SPACING
ONE_WAY = MIRROR_SPACING / C
ROUND_TRIP = 2 * ONE_WAY
TO_FIRST = -SOURCE_X / C
TO_SECOND = TO_FIRST + ONE_WAY
FIRST_LEAK = TO_FIRST + ROUND_TRIP
RISE = 0.45                # 光源振幅用半个余弦平滑升至 1
RESONANCE_ORDER = 3        # 共振时腔长正好容纳 3 个半波长
FRAME_STEP = ONE_WAY / 48  # 一帧推进的模型时间，用于慢放
STEADY_TOL = 1e-6         # 跳到稳态时，首个未到达项相对首束的振幅上界
BEAM_RESONANCE_ORDER = 120  # 有限光束实验使用较短的波长，便于实际形成窄光束
RING_RESONANCE_ORDER = 300  # 让可见的径向视场内包含多阶圆环
SOURCE_PLANE = -2.42 * MIRROR_SPACING
DIVERGING_LENS_X = -1.05 * MIRROR_SPACING
DIVERGING_FOCAL_LENGTH = -0.64 * MIRROR_SPACING
FOCUSING_LENS_X = 1.92 * MIRROR_SPACING
FOCUSING_FOCAL_LENGTH = 1.40 * MIRROR_SPACING
SCREEN_X = FOCUSING_LENS_X + FOCUSING_FOCAL_LENGTH


def mirror_amplitudes(reflectance):
    return math.sqrt(reflectance), 1j * math.sqrt(1 - reflectance)


class FabryPerot:
    """以从光源出发的总路程计算每条光路的相位和到达时间。"""

    def __init__(self, reflectance, detuning):
        self.R = reflectance
        self.r, self.t_m = mirror_amplitudes(reflectance)
        self.omega = (2 * math.pi * RESONANCE_ORDER + detuning) / ROUND_TRIP
        # 一次额外往返包含两次反射和一段长为 2e 的传播。
        self.q = self.r**2 * self.phase(ROUND_TRIP)
        trips = math.ceil(math.log(STEADY_TOL) / math.log(self.R)) if 0 < self.R < 1 else 1
        self.end_time = FIRST_LEAK + trips * ROUND_TRIP + RISE

    @staticmethod
    def envelope(age):
        """光源开启前为零，随后在 RISE 时间内平滑升至单位振幅。"""
        progress = np.clip(age / RISE, 0.0, 1.0)
        return np.sin(0.5 * np.pi * progress)**2

    def phase(self, delay):
        # 长度 L 对应延迟 L/c。在 exp(−iωt) 约定下，延迟给出 exp(+iωL/c)。
        return np.exp(1j * self.omega * delay)

    def path(self, time, delay):
        return self.phase(delay) * self.envelope(np.asarray(time) - delay)

    def path_sum(self, time, first_delay, first_coefficient):
        """求同类光路之和，后续每束比前一束多走一个往返。"""
        age = np.asarray(time) - first_delay
        if first_coefficient == 0:
            return np.zeros_like(age, dtype=complex)

        # 已完全到达的光束构成有限几何级数。启动时间短于往返时间，
        # 因此至多还有一束处于启动过程中，其后的所有光束均未到达。
        complete = np.maximum(0, np.floor((age - RISE) / ROUND_TRIP).astype(int) + 1)
        next_factor = self.q**complete
        arrived = (1 - next_factor) / (1 - self.q)
        arriving = next_factor * self.envelope(age - complete * ROUND_TRIP)
        return first_coefficient * self.phase(first_delay) * (arrived + arriving)

    def input_side(self, x, time):
        incident = self.path(time, (x - SOURCE_X) / C)
        direct = self.r * self.path(time, TO_FIRST - x / C)
        leakage = self.path_sum(time, FIRST_LEAK - x / C, self.t_m**2 * self.r)
        return incident, direct, leakage

    def cavity(self, x, time):
        forward = self.path_sum(time, TO_FIRST + x / C, self.t_m)
        backward = self.path_sum(time, FIRST_LEAK - x / C, self.t_m * self.r)
        return forward, backward

    def transmitted(self, x, time):
        return self.path_sum(time, TO_FIRST + x / C, self.t_m**2)

    def powers(self, time):
        # 功率在镜 1 外侧和镜 2 外侧取样。同向各束先相加再取模平方。
        incident, direct, leakage = self.input_side(0.0, time)
        return (np.abs(incident)**2, np.abs(direct + leakage)**2,
                np.abs(self.transmitted(MIRROR_SPACING, time))**2)

    def returning_phasors(self, time, count=4):
        # 去掉公共的 exp(−iωt) 和光源到镜 1 的相位，使入射参考方向朝右。
        reference = self.phase(TO_FIRST)
        _, direct, leakage = self.input_side(0.0, time)
        index = np.arange(count)
        delays = FIRST_LEAK + index * ROUND_TRIP
        beams = self.t_m**2 * self.r * (self.r**2)**index * self.path(time, delays)
        return direct / reference, leakage / reference, beams / reference

    def stored_energy(self, time):
        # 左、右行波的电场交叉项与磁场交叉项抵消。用中点法积分总能量密度。
        x = MIRROR_SPACING * (np.arange(480) + 0.5) / 480
        forward, backward = self.cavity(x, time)
        return ONE_WAY * np.mean(np.abs(forward)**2 + np.abs(backward)**2)


class FiniteBeam:
    """平行无限大镜面、均匀介质中的稳态标量角谱模型。"""

    def __init__(self, reflectance, wavenumber, angle, width):
        self.R = reflectance
        self.r, self.t_m = mirror_amplitudes(reflectance)
        self.k = wavenumber
        self.angle = angle
        self.width = width
        self.ky_center = self.k * math.sin(angle)
        self.shift = 2 * MIRROR_SPACING * math.tan(angle)

    def transfers(self, ky):
        """返回法向波数，以及漏回、总反射、首束透射、总透射的复系数。"""
        kx = np.sqrt(np.maximum(0.0, (self.k - ky) * (self.k + ky)))
        phase = np.exp(1j * kx * MIRROR_SPACING)
        first = self.t_m**2 * phase
        if self.R == 1:
            return kx, first, np.ones_like(first), first, first
        trip = phase**2
        denominator = 1 - self.R * trip
        leakage = self.t_m**2 * self.r * trip / denominator
        return kx, leakage, self.r + leakage, first, first / denominator

    def near_fields_and_power(self):
        # 镜 1 处的包络是 exp[−(y/w)²]。在角谱网格上用同一组系数计算镜外场与功率。
        # 网格按镜面线宽加密，并将角谱零填充以提高画图时的横向采样密度。
        half_band = 20 / self.width
        resolution = max(0.01, 1 - self.R) / (8 * MIRROR_SPACING)
        count = 2**math.ceil(math.log2(max(1024, 2 * half_band / resolution)))
        index = np.arange(-count // 2, count // 2)
        offset = index * (2 * half_band / count)
        ky = self.ky_center + offset
        spectrum = np.exp(-(0.5 * self.width * offset)**2) * (np.abs(ky) < self.k)
        active = np.abs(offset) <= 8 / self.width
        kx, leakage, reflected, first, transmitted = self.transfers(ky[active])
        spectrum_active = spectrum[active]

        # 每个传播分量穿过镜面的功率包含方向余弦 kx/k。积分覆盖全部出射光。
        weight = (kx / self.k) * spectrum_active**2
        reflected_power = np.dot(weight, np.abs(reflected)**2) / np.sum(weight)
        transmitted_power = np.dot(weight, np.abs(transmitted)**2) / np.sum(weight)

        # 画图窗口包含入射光斑和前几束的几何位置，功率积分不受窗口范围限制。
        y = index * (np.pi / half_band)
        view = (y >= -3 * self.width) & (y <= 3 * self.width + 3.5 * self.shift)
        normalization = count / np.sum(spectrum_active)
        buffer = np.zeros(count, dtype=complex)
        fields = []
        for coefficient in (self.r, leakage, reflected, first, transmitted):
            buffer[active] = spectrum_active * coefficient
            field = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(buffer)))
            fields.append(normalization * np.abs(field[view]))
        return y[view], fields, reflected_power, transmitted_power

    def angular_view(self):
        # 画出中心附近最多约一个自由光谱范围，便于看到共振处的窄峰和暗点。
        half_width = min(3 / self.width, np.pi / (2 * MIRROR_SPACING * max(0.05, math.tan(self.angle))))
        offset = np.linspace(-half_width, half_width, 1601)
        ky = self.ky_center + offset
        angles = np.arcsin(ky / self.k) - self.angle
        _, _, reflected, _, transmitted = self.transfers(ky)
        incoming = np.exp(-0.5 * (self.width * offset)**2)
        return np.rad2deg(angles), incoming, np.abs(reflected)**2, np.abs(transmitted)**2


def airy_coefficient(reflectance):
    return 4 * reflectance / (1 - reflectance)**2


def airy_transmission(phase, coefficient):
    return 1 / (1 + coefficient * np.sin(0.5 * phase)**2)


def averaged_airy_transmission(phase, phase_width, coefficient):
    """计算像素跨过的相位区间内的平均透射率，保留高反射率下的细环。"""
    root = math.sqrt(1 + coefficient)

    def integral(value):
        period = np.floor((value + np.pi) / (2 * np.pi))
        local_phase = value - 2 * np.pi * period
        return 2 / root * (np.pi * period + np.arctan(root * np.tan(local_phase / 2)))

    area = integral(phase + phase_width / 2) - integral(phase - phase_width / 2)
    value = np.array(airy_transmission(phase, coefficient))
    return np.divide(area, phase_width, out=value, where=phase_width > 1e-8)


def screen_pattern(reflectance, axis_phase, cone_slope, radius, pixel_width):
    # 理想会聚透镜把相同倾角映射到 r = f tan θ。radius 以光锥在屏上的半径为单位。
    if reflectance == 1:
        return np.zeros_like(radius)
    slope = radius * cone_slope
    norm = np.sqrt(1 + slope**2)
    phase = axis_phase / norm
    phase_width = axis_phase * cone_slope * slope / norm**3 * pixel_width
    transmission = averaged_airy_transmission(phase, phase_width, airy_coefficient(reflectance))
    return np.clip(transmission, 0, 1)


def optical_train_ray(slope, round_trips=0):
    """跟踪从平行光源出发、经薄透镜、往返腔和焦平面的一条子光束。"""
    e = MIRROR_SPACING
    x = np.array([SOURCE_PLANE, DIVERGING_LENS_X, 0, *([e, 0] * round_trips), e, FOCUSING_LENS_X])
    # 发散透镜前的光平行于光轴。之后每段腔内传播都沿横向积累位移。
    y = -DIVERGING_FOCAL_LENGTH * slope + slope * np.r_[0, 0, np.cumsum(np.abs(np.diff(x[1:])))]
    outgoing_slope = slope - y[-1] / FOCUSING_FOCAL_LENGTH
    screen_y = y[-1] + FOCUSING_FOCAL_LENGTH * outgoing_slope
    return np.r_[x, SCREEN_X], np.r_[y, screen_y]


BG = "#f3f6fa"
INK = "#172c45"
MUTED = "#4c6278"
BORDER = "#dbe4ed"
GRID = "#e8edf3"
BLUE = "#1d65bc"
ORANGE = "#bb5020"
GREEN = "#007e69"
PURPLE = "#982476"
TRANSMISSION = "#006f86"
LEAK_COLORS = ("#006c5a", "#00816c", "#008e79", "#329e90", MUTED)
RING_COLOR = "#9e4a12"
RING_COLORMAP = LinearSegmentedColormap.from_list(
    "干涉圆环", ("#080507", "#2f100a", "#8c2310", "#e9630b", "#ffbd47", "#fff5b5"))


# 中文作为原文。每对译文按英语、法语排列。
TRANSLATIONS = {
    "Fabry–Perot 干涉仪": ("Fabry–Perot interferometer", "Interféromètre de Fabry–Perot"),
    "一维正入射  ·  镜距 e = 1，光速 c = 1  ·  光源位于 x/e = −1，腔内往返用时 2": (
        "Normal incidence in 1D  ·  e = c = 1  ·  Source at x/e = −1  ·  Round trip takes 2",
        "Incidence normale en 1D  ·  e = c = 1  ·  Source à x/e = −1  ·  Aller-retour en 2 unités"),
    "斜入射与有限光束": ("Angle and beam width", "Angle et largeur du faisceau"),
    "透射曲线与干涉圆环": ("Transmission and rings", "Transmission et anneaux"),
    "瞬时电场  E(x, t)": ("Instantaneous field E(x, t)", "Champ instantané E(x, t)"),
    "位置  x/e": ("Position x/e", "Position x/e"),
    "归一化电场": ("Normalized field", "Champ normalisé"),
    "入射侧": ("Incident side", "Côté incident"),
    "腔内": ("Cavity", "Cavité"),
    "透射侧": ("Transmission side", "Côté transmis"),
    "镜 1": ("Mirror 1", "Miroir 1"),
    "镜 2": ("Mirror 2", "Miroir 2"),
    "入射波 →": ("Incident wave →", "Onde incidente →"),
    "直接反射 ←": ("Direct reflection ←", "Réflexion directe ←"),
    "漏光合波 ←": ("Cavity leakage ←", "Fuite de cavité ←"),
    "总反射 ←": ("Total reflection ←", "Réflexion totale ←"),
    "总电场 E": ("Total field E", "Champ total E"),
    "入射侧返回波的相量": ("Reflected phasors", "Phaseurs réfléchis"),
    "取镜 1 外侧为观察点，入射相量指向实轴正向。": (
        "At mirror 1, the incident phasor points along +Re.", "Au miroir 1, le phaseur incident pointe vers +Re."),
    "漏光首尾相接": ("Leakage tip to tail", "Fuites bout à bout"),
    "返回波相加": ("Total reflection", "Somme réfléchie"),
    "实": ("Re", "Ré"),
    "虚": ("Im", "Im"),
    "前 4 束漏光为绿色，其余合量为灰色。◆ 为总反射。": (
        "First 4 returns in green, rest in gray. ◆ Total reflection.",
        "4 premières fuites en vert, reste en gris. ◆ Somme réfléchie."),
    "镜面外侧的功率": ("Power at the mirrors", "Puissance aux miroirs"),
    "时间  t / (e/c)    ·    t = 1 入射到达，t = 2 开始透射，t = 3 首束漏回": (
        "Time t/(e/c)  ·  t = 1 reaches M1  ·  t = 2 first transmission  ·  t = 3 first return",
        "Temps t/(e/c)  ·  t = 1 arrivée à M1  ·  t = 2 première transmission  ·  t = 3 premier retour"),
    "归一化功率": ("Normalized power", "Puissance normalisée"),
    "入射": ("Incident", "Incident"),
    "总反射": ("Total reflected", "Total réfléchi"),
    "总透射": ("Total transmitted", "Total transmis"),
    "建立时  dU/dt = 入射 − 反射 − 透射。稳态时出射功率之和等于入射。": (
        "During build-up, dU/dt = incident − reflected − transmitted. Steady output equals input.",
        "En transitoire, dU/dt = incidente − réfléchie − transmise. En régime établi, sortie = entrée."),
    "光路与波前": ("Ray paths and wavefronts", "Trajets et fronts d’onde"),
    "镜距 e": ("Gap e", "Écart e"),
    "直接反射": ("Direct", "Directe"),
    "透射 {n}": ("Out {n}", "Sortie {n}"),
    "漏回 {n}": ("Return {n}", "Fuite {n}"),
    "展示前 3 束透射和漏回。彩色箭头随波前推进。": (
        "First 3 beams on each side. Color follows the wavefront.",
        "Trois faisceaux par côté. La couleur suit le front d’onde."),
    "纵向错开便于追踪，正入射的各束光实际沿同一条轴。": (
        "Vertical offsets separate paths that share one axis.", "Le décalage vertical sépare les trajets d’un même axe."),
    "功率反射率 R": ("Power reflectance R", "Réflectance R"),
    "往返失谐 Δφ/π": ("Round-trip detuning Δφ/π", "Écart de phase Δφ/π"),
    "观察时刻": ("Observation time", "Instant observé"),
    "播放": ("Play", "Lecture"),
    "暂停": ("Pause", "Pause"),
    "单步": ("Step", "Pas à pas"),
    "重播": ("Replay", "Rejouer"),
    "跳到稳态": ("Jump to steady state", "Aller au régime établi"),
    "拖动时间滑块可暂停定位。更改反射率或失谐会回到光源开启时刻。": (
        "Drag the time slider to pause and scrub. Changing reflectance or detuning restarts at switch-on.",
        "Déplacer le curseur de temps met en pause. Modifier la réflectance ou la phase relance l’allumage."),
    "振幅  直接反射 {direct:.3f}   漏光合波 {leak:.3f}   总反射 {net:.3g}": (
        "Amplitude  Direct {direct:.3f}   Leakage {leak:.3f}   Total {net:.3g}",
        "Amplitude  Directe {direct:.3f}   Fuite {leak:.3f}   Totale {net:.3g}"),
    "完全反射的镜面把光留在入射侧，腔内没有漏光。": (
        "Perfect mirrors keep all light on the incident side.", "Les miroirs parfaits renvoient toute la lumière incidente."),
    "完全透射的镜面不产生反射，腔内没有往返光束。": (
        "Transparent mirrors pass the beam straight through.", "Les miroirs transparents laissent passer le faisceau."),
    "首束漏光要等镜 1 到镜 2 再返回的一个往返。": (
        "The first leakage arrives after one cavity round trip.", "La première fuite arrive après un aller-retour dans la cavité."),
    "总反射趋近零，蓝色入射波仍朝镜 1 传播。": (
        "Reflection vanishes. The blue incident wave still advances.", "La réflexion s’annule. L’onde incidente bleue avance."),
    "共振时各束漏光同向，漏光合波逐渐抵消直接反射。": (
        "At resonance, in-phase leakage cancels direct reflection.", "À la résonance, les fuites en phase annulent la réflexion directe."),
    "偏离共振时，相邻漏光的夹角为往返失谐 Δφ。": (
        "Away from resonance, successive returns differ by Δφ.", "Hors résonance, deux retours successifs sont déphasés de Δφ."),
    "当前  入射 {incident:.3f}   反射 {reflected:.3g}   透射 {transmitted:.3f}   腔内储能 U = {energy:.3f}": (
        "Now  Incident {incident:.3f}   Reflected {reflected:.3g}   Transmitted {transmitted:.3f}   Stored U = {energy:.3f}",
        "À cet instant  Incidente {incident:.3f}   Réfléchie {reflected:.3g}   Transmise {transmitted:.3f}   Stockée U = {energy:.3f}"),
    "Fabry–Perot 斜入射与有限光束": ("Fabry–Perot · Angle and beam width", "Fabry–Perot · Angle et largeur du faisceau"),
    "稳态高斯光束  ·  w 是镜 1 上电场降到峰值约 37% 的半宽  ·  入射角从镜面法线量起": (
        "Steady Gaussian beam  ·  w is the 37% field radius on mirror 1  ·  Angles are measured from the normal",
        "Faisceau gaussien stationnaire  ·  w est le rayon à 37 % du champ sur M1  ·  Angles mesurés depuis la normale"),
    "反射光斑 · 镜 1 外": ("Reflected field · Outside mirror 1", "Champ réfléchi · Extérieur de M1"),
    "透射光斑 · 镜 2 外": ("Transmitted field · Outside mirror 2", "Champ transmis · Extérieur de M2"),
    "沿镜面的位置  y/e": ("Position along mirror y/e", "Position sur le miroir y/e"),
    "场振幅 |E| / 入射峰值": ("Field |E| / incident peak", "Champ |E| / pic incident"),
    "漏光合量": ("Leakage sum", "Somme des fuites"),
    "首束透射": ("First transmitted beam", "Premier faisceau transmis"),
    "焦平面局部 · 各角度的反射与透射": ("Focal plane · Reflection and transmission", "Plan focal · Réflexion et transmission"),
    "角谱强度 / 入射峰值": ("Angular spectrum / incident peak", "Spectre angulaire / pic incident"),
    "反射": ("Reflected", "Réfléchi"),
    "透射": ("Transmitted", "Transmis"),
    "几何光路 · 光束中心的横向错位": ("Ray geometry · Beam displacement", "Trajets optiques · Décalage des faisceaux"),
    "入射角（°）": ("Incidence (°)", "Incidence (°)"),
    "光束半宽 w/e": ("Beam half-width w/e", "Demi-largeur w/e"),
    "镜面反射率 R": ("Mirror reflectance R", "Réflectance R"),
    "波长模式 · 锁定中心共振": ("Wavelength · Center locked", "Longueur d’onde · Résonance"),
    "波长模式 · 固定波长": ("Wavelength · Fixed", "Longueur d’onde · Fixe"),
    "振幅刻度 · 线性": ("Amplitude scale · Linear", "Échelle du champ · Linéaire"),
    "振幅刻度 · 对数": ("Amplitude scale · Log", "Échelle du champ · Log"),
    "总功率收集全部出射光。光斑图放大入射位置及前几束。点角度数值可精确输入。": (
        "Power totals include all outgoing light. Field plots show the first few spots. Click the angle value to type it.",
        "Les puissances incluent toute la lumière sortante. Les champs montrent les premières taches. Cliquer sur l’angle pour le saisir."),
    "相对中心入射角（毫度）": ("Angle from beam center (millidegrees)", "Écart à l’angle central (millidegrés)"),
    "相对中心入射角（°）": ("Angle from beam center (°)", "Écart à l’angle central (°)"),
    "总反射 {reflected:.2%}    总透射 {transmitted:.2%}    中心角透射 {center:.2%}": (
        "Total reflected {reflected:.2%}    Total transmitted {transmitted:.2%}    At center angle {center:.2%}",
        "Réflexion totale {reflected:.2%}    Transmission totale {transmitted:.2%}    À l’angle central {center:.2%}"),
    "波长 λ/e = {wavelength:.5f}    中心失谐 {detuning:+.3f}π    每次往返横移 Δy/e = {shift:.3f}    Δy/w = {ratio:.3g}": (
        "Wavelength λ/e = {wavelength:.5f}    Center detuning {detuning:+.3f}π    Shift per round trip Δy/e = {shift:.3f}    Δy/w = {ratio:.3g}",
        "Longueur d’onde λ/e = {wavelength:.5f}    Écart central {detuning:+.3f}π    Décalage par aller-retour Δy/e = {shift:.3f}    Δy/w = {ratio:.3g}"),
    "Fabry–Perot 透射曲线与干涉圆环": ("Fabry–Perot · Transmission and rings", "Fabry–Perot · Transmission et anneaux"),
    "单色光经发散透镜照亮平行腔镜。会聚透镜将相同倾角的光汇到屏上同一个半径。": (
        "A diverging lens illuminates the cavity. The focusing lens maps equal angles to the same screen radius.",
        "Une lentille divergente éclaire la cavité. La lentille convergente associe un rayon sur l’écran à chaque angle."),
    "发散半角（°）": ("Half-angle (°)", "Demi-angle (°)"),
    "中心失谐 Δφ/π": ("Axial detuning Δφ/π", "Écart axial Δφ/π"),
    "理想薄透镜与平行无损腔镜。两张屏图使用相同亮度色标，屏位于会聚透镜的后焦面。": (
        "Ideal thin lenses and lossless parallel mirrors. Both images share a brightness scale. The screen is at the rear focal plane.",
        "Lentilles minces idéales et miroirs parallèles sans pertes. Même échelle de luminosité pour les deux images au foyer image."),
    "艾里透射曲线": ("Airy transmission curves", "Courbes de transmission d’Airy"),
    "相对往返相位  φ − 2pπ": ("Relative round-trip phase φ − 2pπ", "Phase relative d’aller-retour φ − 2pπ"),
    "透射率": ("Transmittance", "Transmission"),
    "当前镜面": ("Current mirrors", "Miroirs actuels"),
    "峰宽由 F = 4R/(1−R)² 决定。每增加 2π，透射曲线重复一次。": (
        r"$\mathcal{F}=4R/(1-R)^2$ sets the peak width. The period is $2\pi$.",
        r"$\mathcal{F}=4R/(1-R)^2$ fixe la largeur des pics. La période vaut $2\pi$."),
    "高斯照明": ("Gaussian illumination", "Éclairage gaussien"),
    "均匀照明": ("Uniform illumination", "Éclairage uniforme"),
    "总光路": ("Full optical path", "Trajet optique complet"),
    "青线为选中角度，橙色虚线多绕一圈，在焦平面汇到同一点。": (
        "Teal marks the selected angle. The dashed ray makes one extra round trip and meets it at the screen.",
        "En bleu-vert, l’angle choisi. Le rayon en pointillés fait un aller-retour de plus et rejoint le même point sur l’écran."),
    "平行单色光": ("Collimated light", "Lumière collimatée"),
    "发散透镜": ("Diverging lens", "Lentille divergente"),
    "会聚透镜": ("Focusing lens", "Lentille convergente"),
    "焦平面屏": ("Focal-plane screen", "Écran au foyer"),
    "后焦距 f": ("Rear focal length f", "Distance focale image f"),
    "点击圆环取样    θ = {angle:.2f}°    该角度透射率 {transmission:.2%}": (
        "Click a ring to sample    θ = {angle:.2f}°    Transmittance {transmission:.2%}",
        "Cliquer sur un anneau    θ = {angle:.2f}°    Transmission {transmission:.2%}"),
}


class LanguageSettings:
    """保存本次运行的界面语言，并通知同一程序中的各窗口。"""

    def __init__(self):
        self.code = "en"
        self.windows = WeakSet()

    def text(self, message, **values):
        translations = TRANSLATIONS.get(message)
        if translations is not None and self.code != "zh":
            message = translations[0 if self.code == "en" else 1]
        return message.format(**values) if values else message

    def select(self, code):
        if code != self.code:
            self.code = code
            for window in list(self.windows):
                if window.fig.canvas.manager is None:
                    self.windows.discard(window)
                else:
                    window._refresh_language()


@dataclass
class RaySegment:
    """一段示意光路。起点和终点的实部是物理位置，虚部用于错开光束。"""

    arrow: FancyArrowPatch
    start: complex
    end: complex
    delay: float
    reflections: int
    transmissions: int


def configure_font():
    """用拉丁字体显示英法文，并从系统字体中补齐中文字形。"""
    candidates = ("Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Noto Sans SC",
                  "Source Han Sans SC", "WenQuanYi Micro Hei", "SimHei", "Arial Unicode MS")
    installed = set(font_manager.get_font_names())
    family = next((name for name in candidates if name in installed), None)
    fonts = ["DejaVu Sans"] + ([family] if family else [])
    plt.rcParams.update({"font.family": fonts, "axes.unicode_minus": False,
                         "toolbar": "None"})


class PlotWindow:
    def __init__(self, title, language=None):
        self.fig = plt.figure(figsize=(12.6, 7.2), facecolor=BG)
        self.language = language if language is not None else LanguageSettings()
        self.window_title = title
        self.localized_text = {}
        self.language_buttons = {}
        self.language.windows.add(self)
        self.fig.canvas.mpl_connect("close_event", self._unregister_language)
        for index, (code, label) in enumerate((("en", "English"), ("fr", "Français"), ("zh", "中文"))):
            button = self._button([0.752 + index * 0.068, 0.965, 0.065, 0.027], label)
            button.on_clicked(lambda _event, selected=code: self.language.select(selected))
            self.language_buttons[code] = button
        self._refresh_language()

    def _unregister_language(self, _event):
        self.language.windows.discard(self)

    def _refresh_language(self):
        self.fig.canvas.manager.set_window_title(self.language.text(self.window_title))
        for artist, (message, values) in self.localized_text.items():
            artist.set_text(self.language.text(message, **values))
        for code, button in self.language_buttons.items():
            active = code == self.language.code
            button.color = "#deedff" if active else "white"
            button.ax.set_facecolor(button.color)
            button.label.set(color=BLUE if active else INK, weight="bold" if active else "normal")
        self.fig.canvas.draw_idle()

    def _message(self, artist, message, **values):
        self.localized_text[artist] = (message, values)
        artist.set_text(self.language.text(message, **values))
        return artist

    def _fig_text(self, x, y, message, **style):
        return self._message(self.fig.text(x, y, "", **style), message)

    def _axis_text(self, ax, x, y, message, **style):
        return self._message(ax.text(x, y, "", **style), message)

    def _title(self, ax, message, **style):
        return self._message(ax.set_title("", **style), message)

    def _axis_labels(self, ax, *, xlabel=None, ylabel=None):
        if xlabel is not None:
            self._message(ax.xaxis.label, xlabel)
        if ylabel is not None:
            self._message(ax.yaxis.label, ylabel)

    def _legend(self, ax, **style):
        for label in ax.legend(**style).get_texts():
            self._message(label, label.get_text())

    def _axes(self, rect):
        ax = self.fig.add_axes(rect, facecolor="white")
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.set_axisbelow(True)
        return ax

    def _button(self, rect, label, color=INK, background="white"):
        ax = self.fig.add_axes(rect)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        button = Button(ax, label, color=background, hovercolor="#e4eef8")
        button.label.set(color=color, fontsize=9)
        self._message(button.label, label)
        return button

    def _slider(self, rect, label, lo, hi, initial, valfmt, step=None):
        ax = self.fig.add_axes(rect, facecolor=BG)
        slider = Slider(ax, label, lo, hi, valinit=initial, valfmt=valfmt, valstep=step,
                        color=BLUE, track_color="#dce6f0", initcolor="none")
        slider.label.set(color=INK, fontsize=9)
        self._message(slider.label, label)
        slider.valtext.set(color=MUTED, fontsize=9)
        slider.drawon = False
        return slider

    @staticmethod
    def _new_arrow(ax, color):
        arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", shrinkA=0, shrinkB=0,
                                mutation_scale=9, color=color, lw=1.8, zorder=4)
        ax.add_patch(arrow)
        return arrow

    @staticmethod
    def _point_arrow(arrow, start, end):
        arrow.set_positions((start.real, start.imag), (end.real, end.imag))
        arrow.set_visible(abs(end - start) > 1e-8)


class Lesson(PlotWindow):
    """用同一个物理模型更新空间波形、相量、功率和光路波前。"""

    def __init__(self):
        super().__init__("Fabry–Perot 干涉仪")
        self.model = FabryPerot(0.70, 0.0)
        self.time = 0.0
        self.playing = False
        self.moving_time_slider = False
        self.study_windows = {}
        self._fig_text(0.065, 0.956, "Fabry–Perot 干涉仪", fontsize=24, weight="bold", color=INK)
        self._fig_text(0.067, 0.920,
                       "一维正入射  ·  镜距 e = 1，光速 c = 1  ·  光源位于 x/e = −1，腔内往返用时 2",
                       fontsize=10, color=MUTED)

        self._build_wave_plot()
        self._build_phasor_plots()
        self._build_power_plot()
        self._build_ray_plot()
        self._build_controls()
        self.beam_button = self._button([0.597, 0.916, 0.174, 0.043],
                                        "斜入射与有限光束", BLUE, "#eaf3fe")
        self.rings_button = self._button([0.779, 0.916, 0.174, 0.043],
                                         "透射曲线与干涉圆环", BLUE, "#eaf3fe")
        self.beam_button.on_clicked(lambda _event: self._open_study(FiniteBeamLesson))
        self.rings_button.on_clicked(lambda _event: self._open_study(AiryRingLesson))
        self.timer = self.fig.canvas.new_timer(interval=36)
        self.timer.add_callback(self._tick)
        self.fig.canvas.mpl_connect("close_event", lambda _event: self.timer.stop())
        self._set_power_data()
        self._draw_time()

    def _open_study(self, study_type):
        self._set_playing(False)
        self.fig.canvas.draw_idle()
        study = self.study_windows.get(study_type)
        if study is None or study.fig.canvas.manager is None or not plt.fignum_exists(study.fig.number):
            study = study_type(self.model.R, self.language)
            self.study_windows[study_type] = study
            study.fig.canvas.mpl_connect("close_event", lambda _event: self.study_windows.pop(study_type, None))
        study.fig.show()

    def _build_wave_plot(self):
        self._fig_text(0.065, 0.866, "瞬时电场  E(x, t)", fontsize=15, weight="bold", color=INK)
        ax = self._axes([0.065, 0.506, 0.542, 0.294])
        self.wave_ax = ax
        ax.set_xlim(SOURCE_X, 2 * MIRROR_SPACING)
        self._axis_labels(ax, xlabel="位置  x/e", ylabel="归一化电场")
        ax.set_xticks(np.arange(-1, 2.01, 0.5) * MIRROR_SPACING)
        ax.axvspan(0, MIRROR_SPACING, color="#edf5fd", zorder=0)
        ax.grid(axis="y", color=GRID)
        ax.axhline(0, color=BORDER, lw=0.9)
        for position in (0, MIRROR_SPACING):
            ax.axvline(position, color=INK, lw=1.6, zorder=3)

        for x, label in ((-0.5, "入射侧"), (0.5, "腔内"), (1.5, "透射侧"),
                         (0, "镜 1"), (1, "镜 2")):
            self._fig_text(0.065 + 0.542 * (x + 1) / 3, 0.810, label,
                           ha="center", fontsize=9, color=INK if x in (0, 1) else MUTED)
        # 在空间轴顶部标出右行和左行方向，箭头的高度独立于电场纵轴。
        for start, end, color in ((-0.73, -0.47, BLUE), (-0.23, -0.47, GREEN),
                                  (0.22, 0.47, BLUE), (0.78, 0.53, GREEN),
                                  (1.36, 1.62, BLUE)):
            ax.annotate("", xy=(end * MIRROR_SPACING, 0.92), xytext=(start * MIRROR_SPACING, 0.92),
                        xycoords=ax.get_xaxis_transform(),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1))

        self.left_x = np.linspace(SOURCE_X, 0.0, 420)
        self.cavity_x = np.linspace(0.0, MIRROR_SPACING, 420)
        self.right_x = np.linspace(MIRROR_SPACING, 2 * MIRROR_SPACING, 420)

        def line(x, color, style, width, zorder):
            return ax.plot(x, np.zeros_like(x), color=color, ls=style, lw=width,
                           zorder=zorder)[0]

        self.incident_line = line(self.left_x, BLUE, "-", 1.65, 5)
        self.direct_line = line(self.left_x, ORANGE, "--", 1.45, 6)
        self.leak_line = line(self.left_x, GREEN, "--", 1.65, 6)
        self.reflected_line = line(self.left_x, PURPLE, ":", 2.45, 7)
        self.cavity_forward_line = line(self.cavity_x, BLUE, "-", 1.65, 5)
        self.cavity_backward_line = line(self.cavity_x, GREEN, "--", 1.65, 6)
        self.transmitted_line = line(self.right_x, BLUE, "-", 1.7, 5)
        self.total_lines = [line(x, INK, "-", 2.3, 4)
                            for x in (self.left_x, self.cavity_x, self.right_x)]

        # 开关只控制显示。透射侧只有右行波，所以它也就是该侧的总电场。
        components = (
            ("入射波 →", BLUE, [self.incident_line]),
            ("直接反射 ←", ORANGE, [self.direct_line]),
            ("漏光合波 ←", GREEN, [self.leak_line]),
            ("总反射 ←", PURPLE, [self.reflected_line]),
            ("总电场 E", INK, self.total_lines),
        )
        self.component_buttons = []
        for index, (label, color, lines) in enumerate(components):
            rect = [0.065 + index * 0.109, 0.831, 0.102, 0.027]
            button = self._button(rect, label, color)
            button.on_clicked(lambda _event, b=button, group=lines, c=color:
                              self._toggle_component(b, group, c))
            self.component_buttons.append(button)

    def _build_phasor_plots(self):
        self._fig_text(0.652, 0.866, "入射侧返回波的相量", fontsize=15, weight="bold", color=INK)
        self._fig_text(0.653, 0.834, "取镜 1 外侧为观察点，入射相量指向实轴正向。",
                       fontsize=9, color=MUTED)
        self.leaks_ax = self._axes([0.652, 0.563, 0.145, 0.235])
        self.sum_ax = self._axes([0.808, 0.563, 0.145, 0.235])
        for ax, title in ((self.leaks_ax, "漏光首尾相接"), (self.sum_ax, "返回波相加")):
            ax.set(aspect="equal", xlim=(-1.24, 1.24), ylim=(-1.24, 1.24), xticks=[], yticks=[])
            ax.axhline(0, color=BORDER, lw=1)
            ax.axvline(0, color=BORDER, lw=1)
            self._axis_text(ax, 0.5, 0.04, title, transform=ax.transAxes, ha="center", va="bottom",
                            fontsize=9, color=MUTED)
            self._axis_text(ax, 0.96, 0.53, "实", transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
            self._axis_text(ax, 0.53, 0.94, "虚", transform=ax.transAxes, va="top", fontsize=8, color=MUTED)

        self.leak_arrows = [self._new_arrow(self.leaks_ax, color) for color in LEAK_COLORS]
        self.direct_arrow = self._new_arrow(self.sum_ax, ORANGE)
        self.leak_sum_arrow = self._new_arrow(self.sum_ax, GREEN)
        self.net_dot, = self.sum_ax.plot([], [], "D", color=PURPLE, ms=5, zorder=9,
                                         markeredgecolor="white", markeredgewidth=0.75)
        self._fig_text(0.653, 0.539, "前 4 束漏光为绿色，其余合量为灰色。◆ 为总反射。",
                       fontsize=9, color=MUTED)
        self.phasor_readout = self.fig.text(0.653, 0.512, "", fontsize=9, color=INK)
        self.phasor_note = self.fig.text(0.653, 0.485, "", fontsize=9, color=MUTED)

    def _build_power_plot(self):
        self._fig_text(0.065, 0.435, "镜面外侧的功率", fontsize=15, weight="bold", color=INK)
        ax = self._axes([0.065, 0.261, 0.542, 0.155])
        self.power_ax = ax
        ax.set_ylim(-0.06, 1.16)
        self._axis_labels(ax, xlabel="时间  t / (e/c)    ·    t = 1 入射到达，t = 2 开始透射，t = 3 首束漏回",
                          ylabel="归一化功率")
        ax.xaxis.label.set_fontsize(9)
        ax.set_yticks((0.0, 0.5, 1.0))
        ax.grid(color=GRID)
        for time in (TO_FIRST, TO_SECOND, FIRST_LEAK):
            ax.axvline(time, color="#afbfce", ls=":", lw=1)
        self.power_lines = [ax.plot([], [], color=color, ls=style, lw=1.9, label=label)[0]
                            for color, style, label in ((BLUE, "--", "入射"),
                                                        (PURPLE, "-", "总反射"),
                                                        (TRANSMISSION, "-", "总透射"))]
        self._legend(ax, loc="upper right", bbox_to_anchor=(1, 1.18), borderaxespad=0,
                     ncol=3, fontsize=9, frameon=False)
        self.cursor = ax.axvline(0.0, color=INK, lw=1.0, alpha=0.78)
        self.power_readout = self.fig.text(0.065, 0.173, "", fontsize=9, color=INK)
        self._fig_text(0.065, 0.150, "建立时  dU/dt = 入射 − 反射 − 透射。稳态时出射功率之和等于入射。",
                       fontsize=9, color=MUTED)

    def _build_ray_plot(self):
        self._fig_text(0.652, 0.435, "光路与波前", fontsize=15, weight="bold", color=INK)
        ax = self._axes([0.652, 0.218, 0.301, 0.198])
        e = MIRROR_SPACING
        ax.set(xlim=(-1.62 * e, 2.55 * e), ylim=(-1.36, 8), xticks=[], yticks=[])
        ax.axvspan(0, e, color="#edf5fd", zorder=0)
        for x, label in ((0, "镜 1"), (e, "镜 2")):
            ax.axvline(x, color=INK, lw=1.25, zorder=3)
            self._axis_text(ax, x, 7.48, label, ha="center", va="center", fontsize=8, color=INK,
                            bbox=dict(facecolor="white", edgecolor="none", pad=1.1), zorder=5)
        ax.annotate("", xy=(e, 6.82), xytext=(0, 6.82),
                    arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8))
        self._axis_text(ax, e / 2, 7.07, "镜距 e", ha="center", va="bottom", fontsize=8, color=MUTED)

        self.rays = []

        def add_segment(start, end, delay, reflections, transmissions, color):
            ax.plot([start.real, end.real], [start.imag, end.imag], color="#a7b7c8", lw=0.95)
            arrow = self._new_arrow(ax, color)
            self.rays.append(RaySegment(arrow, start, end, delay, reflections, transmissions))

        def label_endpoint(position, label, color, left, **values):
            offset = -4 if left else 4
            artist = ax.annotate("", (position.real, position.imag), xytext=(offset, 0),
                                 textcoords="offset points", ha="right" if left else "left",
                                 va="center", fontsize=8, color=color)
            self._message(artist, label, **values)

        source = complex(SOURCE_X, 7)
        first_hit = complex(0, 6)
        direct_end = complex(-e, 5)
        add_segment(source, first_hit, 0, 0, 0, BLUE)
        add_segment(first_hit, direct_end, TO_FIRST, 1, 0, ORANGE)
        label_endpoint(source, "入射", BLUE, True)
        label_endpoint(direct_end, "直接反射", ORANGE, True)

        # 每一段腔内光路都走一个单程。到达镜面时分成返回腔内和离开腔体的两束。
        for leg in range(6):
            rightward = leg % 2 == 0
            y = 6 - leg
            start = complex(0 if rightward else e, y)
            strike = complex(e if rightward else 0, y - 1)
            out = complex(2 * e if rightward else -e, y - 2)
            delay = TO_FIRST + leg * ONE_WAY
            add_segment(start, strike, delay, leg, 1, BLUE if rightward else GREEN)
            color = TRANSMISSION if rightward else GREEN
            add_segment(strike, out, delay + ONE_WAY, leg, 2, color)
            label = "透射 {n}" if rightward else "漏回 {n}"
            label_endpoint(out, label, color, not rightward, n=leg // 2 + 1)

        self._fig_text(0.652, 0.193, "展示前 3 束透射和漏回。彩色箭头随波前推进。",
                       fontsize=9, color=MUTED)
        self._fig_text(0.652, 0.170, "纵向错开便于追踪，正入射的各束光实际沿同一条轴。",
                       fontsize=9, color=MUTED)

    def _draw_rays(self):
        for ray in self.rays:
            coefficient = self.model.r**ray.reflections * self.model.t_m**ray.transmissions
            # 纵向偏移仅用于展示。传播时间由真实的一维路程 |Δx|/c 决定。
            flight_time = abs(ray.end.real - ray.start.real) / C
            progress = min(1.0, max(0.0, (self.time - ray.delay) / flight_time)) if coefficient else 0.0
            front = ray.start + progress * (ray.end - ray.start)
            self._point_arrow(ray.arrow, ray.start, front)

    def _build_controls(self):
        self.r_slider = self._slider([0.195, 0.115, 0.288, 0.027], "功率反射率 R",
                                     0.0, 1.0, self.model.R, "%.2f", 0.01)
        self.phase_slider = self._slider([0.657, 0.115, 0.264, 0.027], "往返失谐 Δφ/π",
                                         -1.0, 1.0, 0.0, "%+.2f", 0.01)
        self.time_slider = self._slider([0.195, 0.063, 0.283, 0.027], "观察时刻",
                                        0.0, self.model.end_time, 0.0, "%.2f")
        self.play_button = self._button([0.551, 0.056, 0.083, 0.038], "播放", BLUE, "#eaf3fe")
        self.step_button = self._button([0.644, 0.056, 0.075, 0.038], "单步")
        self.replay_button = self._button([0.729, 0.056, 0.075, 0.038], "重播")
        self.steady_button = self._button([0.814, 0.056, 0.139, 0.038], "跳到稳态")

        self.r_slider.on_changed(self._change_model)
        self.phase_slider.on_changed(self._change_model)
        self.time_slider.on_changed(self._seek)
        self.play_button.on_clicked(self._toggle_playing)
        self.step_button.on_clicked(self._step)
        self.replay_button.on_clicked(self._replay)
        self.steady_button.on_clicked(self._jump_to_steady)
        self._fig_text(0.065, 0.018, "拖动时间滑块可暂停定位。更改反射率或失谐会回到光源开启时刻。",
                       fontsize=9, color=MUTED)

    def _toggle_component(self, button, lines, color):
        visible = not lines[0].get_visible()
        for line in lines:
            line.set_visible(visible)
        button.color = "white" if visible else "#e4e9ef"
        button.ax.set_facecolor(button.color)
        button.label.set_color(color if visible else MUTED)
        self._draw_waves()
        self.fig.canvas.draw_idle()

    def _set_power_data(self):
        # 在每个可能的到达时刻采样启动段，并用直线连接中间的平坦段。
        events = np.arange(TO_FIRST, self.model.end_time, ONE_WAY)
        starts = events[:, None] + np.linspace(0, RISE, 16)
        time = np.unique(np.r_[0, np.minimum(starts.ravel(), self.model.end_time), self.model.end_time])
        powers = self.model.powers(time)
        for line, values in zip(self.power_lines, powers):
            line.set_data(time, values)
        self.power_ax.set_ylim(-0.06, max(1.16, 1.08 * np.max(powers)))

    def _change_model(self, _value):
        self._set_playing(False)
        self.model = FabryPerot(self.r_slider.val, np.pi * self.phase_slider.val)
        self.time_slider.valmax = self.model.end_time
        self.time_slider.ax.set_xlim(0, self.model.end_time)
        self._set_power_data()
        self._move_time(0.0)

    def _draw_waves(self):
        incident, direct, leakage = self.model.input_side(self.left_x, self.time)
        forward, backward = self.model.cavity(self.cavity_x, self.time)
        transmitted = self.model.transmitted(self.right_x, self.time)
        reflected = direct + leakage
        fields = ((self.incident_line, incident), (self.direct_line, direct),
                  (self.leak_line, leakage), (self.reflected_line, reflected),
                  (self.cavity_forward_line, forward), (self.cavity_backward_line, backward),
                  (self.transmitted_line, transmitted), (self.total_lines[0], incident + reflected),
                  (self.total_lines[1], forward + backward), (self.total_lines[2], transmitted))
        time_factor = np.exp(-1j * self.model.omega * self.time)
        peak = 1.65
        for line, amplitude in fields:
            line.set_ydata(np.real(amplitude * time_factor))
            if line.get_visible():
                peak = max(peak, np.max(np.abs(amplitude)))
        limit = math.ceil(peak * 2.4) / 2
        if self.wave_ax.get_ylim() != (-limit, limit):
            self.wave_ax.set_ylim(-limit, limit)

    def _draw_phasors(self):
        direct, leakage, beams = self.model.returning_phasors(self.time)
        reflected = direct + leakage
        # 漏光图将各束首尾相接。第五段补上未单独画出的所有光束。
        points = np.r_[0, np.cumsum(beams), leakage]
        for arrow, start, end in zip(self.leak_arrows, points[:-1], points[1:]):
            self._point_arrow(arrow, start, end)
        self._point_arrow(self.direct_arrow, 0j, direct)
        self._point_arrow(self.leak_sum_arrow, 0j, leakage)
        self.net_dot.set_data([reflected.real], [reflected.imag])
        self._message(self.phasor_readout, "振幅  直接反射 {direct:.3f}   漏光合波 {leak:.3f}   总反射 {net:.3g}",
                      direct=abs(direct), leak=abs(leakage), net=abs(reflected))

        if self.model.R == 1:
            note = "完全反射的镜面把光留在入射侧，腔内没有漏光。"
        elif self.model.R == 0:
            note = "完全透射的镜面不产生反射，腔内没有往返光束。"
        elif self.time <= FIRST_LEAK:
            note = "首束漏光要等镜 1 到镜 2 再返回的一个往返。"
        elif self.phase_slider.val == 0 and self.time >= self.model.end_time:
            note = "总反射趋近零，蓝色入射波仍朝镜 1 传播。"
        elif self.phase_slider.val == 0:
            note = "共振时各束漏光同向，漏光合波逐渐抵消直接反射。"
        else:
            note = "偏离共振时，相邻漏光的夹角为往返失谐 Δφ。"
        self._message(self.phasor_note, note)

    def _draw_time(self):
        self._draw_waves()
        self._draw_phasors()
        self._draw_rays()
        self.cursor.set_xdata([self.time, self.time])
        # 先放大最初几次往返，随观察时刻向右移动再扩展功率图的时间范围。
        right = min(self.model.end_time, max(14.0, 5 * math.ceil((self.time + 1) / 5)))
        if self.power_ax.get_xlim() != (0.0, right):
            self.power_ax.set_xlim(0, right)
        incident, reflected, transmitted = self.model.powers(self.time)
        energy = self.model.stored_energy(self.time)
        self._message(self.power_readout,
                      "当前  入射 {incident:.3f}   反射 {reflected:.3g}   透射 {transmitted:.3f}   腔内储能 U = {energy:.3f}",
                      incident=incident, reflected=reflected, transmitted=transmitted, energy=energy)
        self.fig.canvas.draw_idle()

    def _seek(self, time):
        if not self.moving_time_slider:
            self._set_playing(False)
        self.time = time
        self._draw_time()

    def _move_time(self, time):
        self.moving_time_slider = True
        try:
            self.time_slider.set_val(min(time, self.model.end_time))
        finally:
            self.moving_time_slider = False

    def _set_playing(self, playing):
        self.playing = playing
        self._message(self.play_button.label, "暂停" if playing else "播放")
        if playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _tick(self):
        self._move_time(self.time + FRAME_STEP)
        if self.time >= self.model.end_time:
            self._set_playing(False)
            self.fig.canvas.draw_idle()

    def _toggle_playing(self, _event):
        if not self.playing and self.time >= self.model.end_time:
            self._move_time(0.0)
        self._set_playing(not self.playing)
        self.fig.canvas.draw_idle()

    def _step(self, _event):
        self._set_playing(False)
        self._move_time(self.time + FRAME_STEP)

    def _replay(self, _event):
        self._set_playing(True)
        self._move_time(0.0)

    def _jump_to_steady(self, _event):
        self._set_playing(False)
        self._move_time(self.model.end_time)


class FiniteBeamLesson(PlotWindow):
    """用可调角度和束宽比较镜面附近的光斑、角谱和总功率。"""

    def __init__(self, reflectance, language=None):
        super().__init__("Fabry–Perot 斜入射与有限光束", language)
        self.locked = True
        self.log_amplitude = False
        self._fig_text(0.065, 0.956, "斜入射与有限光束", fontsize=24, weight="bold", color=INK)
        self._fig_text(0.065, 0.918,
                       "稳态高斯光束  ·  w 是镜 1 上电场降到峰值约 37% 的半宽  ·  入射角从镜面法线量起",
                       fontsize=10, color=MUTED)
        self.power_text = self.fig.text(0.065, 0.860, "", fontsize=14, weight="bold", color=INK)
        self.phase_text = self.fig.text(0.065, 0.827, "", fontsize=9, color=MUTED)

        self.reflection_ax = self._axes([0.070, 0.542, 0.405, 0.223])
        self.transmission_ax = self._axes([0.552, 0.542, 0.400, 0.223])
        for ax, title in ((self.reflection_ax, "反射光斑 · 镜 1 外"),
                          (self.transmission_ax, "透射光斑 · 镜 2 外")):
            self._title(ax, title, loc="left", fontsize=12, color=INK, pad=10)
            self._axis_labels(ax, xlabel="沿镜面的位置  y/e", ylabel="场振幅 |E| / 入射峰值")
            ax.xaxis.set_major_locator(MaxNLocator(6))
            ax.grid(color=GRID)

        self.near_lines = [ax.plot([], [], color=color, ls=style, lw=width, label=label)[0]
                           for ax, color, style, width, label in (
                               (self.reflection_ax, ORANGE, "--", 2.3, "直接反射"),
                               (self.reflection_ax, GREEN, ":", 1.8, "漏光合量"),
                               (self.reflection_ax, PURPLE, "-", 1.8, "总反射"),
                               (self.transmission_ax, BLUE, "--", 2.0, "首束透射"),
                               (self.transmission_ax, TRANSMISSION, "-", 1.6, "总透射"))]
        for ax in (self.reflection_ax, self.transmission_ax):
            self._legend(ax, loc="upper right", ncol=len(ax.lines), fontsize=8, framealpha=0.92, edgecolor=BORDER)

        self.angular_ax = self._axes([0.070, 0.250, 0.405, 0.182])
        self._title(self.angular_ax, "焦平面局部 · 各角度的反射与透射", loc="left", fontsize=12, color=INK, pad=9)
        self.angular_ax.set(ylim=(-0.03, 1.08), yticks=(0, .5, 1))
        self._axis_labels(self.angular_ax, ylabel="角谱强度 / 入射峰值")
        self.angular_ax.xaxis.set_major_locator(MaxNLocator(6))
        self.angular_ax.grid(color=GRID)
        self.angular_lines = [self.angular_ax.plot([], [], color=color, ls=style, lw=1.6, label=label)[0]
                              for color, style, label in ((BLUE, "--", "入射"), (PURPLE, "-", "反射"),
                                                          (TRANSMISSION, "-", "透射"))]
        self.angular_ax.axvline(0, color=MUTED, lw=0.8, alpha=0.65)
        self._legend(self.angular_ax, loc="upper right", ncol=3, fontsize=8, framealpha=0.92, edgecolor=BORDER)

        self.geometry_ax = self._axes([0.552, 0.250, 0.400, 0.182])
        self._title(self.geometry_ax, "几何光路 · 光束中心的横向错位", loc="left", fontsize=12, color=INK, pad=9)
        e = MIRROR_SPACING
        self.geometry_ax.set_xlim(-0.65 * e, 1.65 * e)
        self._axis_labels(self.geometry_ax, ylabel="沿镜面的位置  y/e")
        self.geometry_ax.set_xticks((0, e))
        mirror_labels = {0: "镜 1", e: "镜 2"}
        self.geometry_ax.xaxis.set_major_formatter(
            lambda position, _index: self.language.text(mirror_labels.get(position, "")))
        self.geometry_ax.yaxis.set_major_locator(MaxNLocator(4))
        self.geometry_ax.axvspan(0, e, color="#edf5fd", zorder=0)
        for x in (0, e):
            self.geometry_ax.axvline(x, color=INK, lw=1.2)
        self.geometry_arrows = [self._new_arrow(self.geometry_ax, color) for color in
                                (BLUE, ORANGE, BLUE, TRANSMISSION, GREEN, GREEN, BLUE, TRANSMISSION)]

        self.angle_slider = self._slider([0.192, 0.130, 0.276, 0.025], "入射角（°）", 0, 45, 25, "%.4f", 0.0001)
        self.angle_slider.valtext.set_visible(False)
        angle_ax = self._axes([0.480, 0.126, 0.066, 0.033])
        self.angle_input = TextBox(angle_ax, "", initial="25.0000", color="white", hovercolor="#eaf3fe",
                                   textalignment="center")
        self.angle_input.text_disp.set(color=INK, fontsize=9)
        self.width_slider = self._slider([0.675, 0.130, 0.243, 0.025], "光束半宽 w/e", -1, 3.3, 3, "%.2f", 0.01)
        self.r_slider = self._slider([0.192, 0.075, 0.276, 0.025], "镜面反射率 R", 0, 1, reflectance, "%.2f", 0.01)
        self.lock_button = self._button([0.552, 0.068, 0.224, 0.044], "波长模式 · 锁定中心共振", BLUE, "#eaf3fe")
        self.scale_button = self._button([0.787, 0.068, 0.165, 0.044], "振幅刻度 · 线性")
        self._fig_text(0.065, 0.024, "总功率收集全部出射光。光斑图放大入射位置及前几束。点角度数值可精确输入。",
                       fontsize=9, color=MUTED)
        for slider in (self.angle_slider, self.width_slider, self.r_slider):
            slider.on_changed(self._refresh)
        self.lock_button.on_clicked(self._toggle_lock)
        self.scale_button.on_clicked(self._toggle_scale)
        self.angle_input.on_submit(self._enter_angle)
        self._refresh(None)

    def _enter_angle(self, text):
        try:
            angle = float(text.replace(",", "."))
        except ValueError:
            angle = self.angle_slider.val
        if not 0 <= angle <= 45:
            angle = self.angle_slider.val
        self.angle_slider.set_val(round(angle, 4))

    def _toggle_lock(self, _event):
        self.locked = not self.locked
        mode = "波长模式 · 锁定中心共振" if self.locked else "波长模式 · 固定波长"
        self._message(self.lock_button.label, mode)
        self._refresh(None)

    def _toggle_scale(self, _event):
        self.log_amplitude = not self.log_amplitude
        for ax in (self.reflection_ax, self.transmission_ax):
            if self.log_amplitude:
                ax.set_yscale("symlog", linthresh=0.001, linscale=0.5)
            else:
                ax.set_yscale("linear")
        scale = "振幅刻度 · 对数" if self.log_amplitude else "振幅刻度 · 线性"
        self._message(self.scale_button.label, scale)
        self.fig.canvas.draw_idle()

    def _draw_geometry(self, model):
        slope = math.tan(model.angle)
        e = MIRROR_SPACING

        def point(x, distance):
            return e * complex(x, slope * distance)

        routes = [(point(-0.52, -0.52), 0j, True), (0j, point(-0.52, 0.52), model.r != 0)]
        for leg in range(3):
            rightward = leg % 2 == 0
            start = point(0 if rightward else 1, leg)
            hit = point(1 if rightward else 0, leg + 1)
            out = point(1.52 if rightward else -0.52, leg + 1.52)
            routes.append((start, hit, bool(model.t_m * model.r**leg)))
            routes.append((hit, out, bool(model.t_m**2 * model.r**leg)))
        for arrow, (start, end, visible) in zip(self.geometry_arrows, routes):
            self._point_arrow(arrow, start, end if visible else start)
        self.geometry_ax.set_ylim(-max(0.1, 0.73 * slope) * e, max(0.1, 3.76 * slope) * e)

    def _refresh(self, _value):
        theta = math.radians(self.angle_slider.val)
        width_ratio = 10**self.width_slider.val
        e = MIRROR_SPACING
        if self.locked:
            # 使中心平面波的法向往返相位 2ke cos θ 等于 2πm。
            self.k = math.pi * BEAM_RESONANCE_ORDER / (e * math.cos(theta))
        model = FiniteBeam(self.r_slider.val, self.k, theta, e * width_ratio)
        y, fields, reflected, transmitted = model.near_fields_and_power()
        for line, magnitude in zip(self.near_lines, fields):
            line.set_data(y / e, magnitude)
        for ax, group in ((self.reflection_ax, fields[:3]), (self.transmission_ax, fields[3:])):
            ax.set_xlim(y[0] / e, y[-1] / e)
            peak = max(np.max(field) for field in group)
            ax.set_ylim(0, max(0.002, 1.28 * peak))

        angle_offsets, incoming, reflection_coefficient, transmission_coefficient = model.angular_view()
        use_millidegrees = angle_offsets[-1] - angle_offsets[0] < 0.12
        angles = angle_offsets * (1000 if use_millidegrees else 1)
        angle_label = "相对中心入射角（毫度）" if use_millidegrees else "相对中心入射角（°）"
        self.angular_ax.set_xlim(angles[0], angles[-1])
        self._axis_labels(self.angular_ax, xlabel=angle_label)
        for line, spectrum in zip(self.angular_lines,
                                  (incoming, incoming * reflection_coefficient, incoming * transmission_coefficient)):
            line.set_data(angles, spectrum)
        self._draw_geometry(model)

        center = len(transmission_coefficient) // 2
        center_phase = 2 * self.k * e * math.cos(theta)
        detuning = math.remainder(center_phase, 2 * math.pi) / math.pi
        detuning = 0.0 if abs(detuning) < 1e-12 else detuning
        self.angle_input.text_disp.set_text(f"{self.angle_slider.val:.4f}")
        self.width_slider.valtext.set_text(f"{width_ratio:.4g}")
        self._message(self.power_text, "总反射 {reflected:.2%}    总透射 {transmitted:.2%}    中心角透射 {center:.2%}",
                      reflected=reflected, transmitted=transmitted, center=transmission_coefficient[center])
        self._message(self.phase_text,
                      "波长 λ/e = {wavelength:.5f}    中心失谐 {detuning:+.3f}π    每次往返横移 Δy/e = {shift:.3f}    Δy/w = {ratio:.3g}",
                      wavelength=2 * math.pi / (self.k * e), detuning=detuning,
                      shift=model.shift / e, ratio=model.shift / model.width)
        self.fig.canvas.draw_idle()


class AiryRingLesson(PlotWindow):
    """把平面波透射率、等倾圆环和整套照明及成像光路对应起来。"""

    def __init__(self, reflectance, language=None):
        super().__init__("Fabry–Perot 透射曲线与干涉圆环", language)
        self._fig_text(0.065, 0.956, "透射曲线与干涉圆环", fontsize=24, weight="bold", color=INK)
        self._fig_text(0.067, 0.918,
                       "单色光经发散透镜照亮平行腔镜。会聚透镜将相同倾角的光汇到屏上同一个半径。",
                       fontsize=10, color=MUTED)
        self._build_phase_plot()
        self._build_ring_views()
        self._build_train()

        self.r_slider = self._slider([0.171, 0.096, 0.189, 0.024], "镜面反射率 R", 0, 1, reflectance, "%.2f", 0.01)
        self.cone_slider = self._slider([0.486, 0.096, 0.172, 0.024], "发散半角（°）", 6, 24, 18, "%.1f", 0.1)
        self.phase_slider = self._slider([0.793, 0.096, 0.127, 0.024], "中心失谐 Δφ/π", -1, 1, 0, "%+.2f", 0.01)
        for slider in (self.r_slider, self.cone_slider, self.phase_slider):
            slider.on_changed(self._refresh)
        self._fig_text(0.065, 0.035,
                       "理想薄透镜与平行无损腔镜。两张屏图使用相同亮度色标，屏位于会聚透镜的后焦面。",
                       fontsize=9, color=MUTED)
        self.fig.canvas.mpl_connect("button_press_event", self._select_angle)
        # 初始跟踪一条离轴共振环，便于看清两束光的分离和会聚。
        ring_slope = math.sqrt((RING_RESONANCE_ORDER / (RING_RESONANCE_ORDER - 4))**2 - 1)
        self.selected = (ring_slope / math.tan(math.radians(self.cone_slider.val)), 0.0)
        self._refresh(None)

    def _build_phase_plot(self):
        ax = self._axes([0.069, 0.530, 0.409, 0.287])
        self._title(ax, "艾里透射曲线", loc="left", fontsize=14, color=INK, pad=12)
        ax.set(xlim=(-0.60, 4.60), ylim=(-0.035, 1.07))
        self._axis_labels(ax, xlabel="相对往返相位  φ − 2pπ", ylabel="透射率")
        ax.set_xticks(np.arange(5), ("0", "π", "2π", "3π", "4π"))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(color=GRID)
        self.phase_grid = np.unique(np.r_[np.linspace(-0.60, 4.60, 8001), (0.0, 2.0, 4.0)])
        for coefficient, color, style in ((10, INK, "-"), (100, ORANGE, "--"), (1000, BLUE, "-.")):
            ax.plot(self.phase_grid, airy_transmission(np.pi * self.phase_grid, coefficient),
                    color=color, ls=style, lw=1.6, label=rf"$\mathcal{{F}}={coefficient}$")
        self.current_curve, = ax.plot([], [], color=PURPLE, ls=":", lw=1.8, label="当前镜面")
        self.phase_cursor = ax.axvline(0.0, color=RING_COLOR, lw=0.95, alpha=0.75)
        self.phase_dot, = ax.plot([], [], "o", ms=5, color=RING_COLOR,
                                 markeredgecolor="white", markeredgewidth=0.7)
        self._legend(ax, loc="center", bbox_to_anchor=(0.306, 0.65), fontsize=8, framealpha=0.94, edgecolor=BORDER)
        self._fig_text(0.069, 0.449, "峰宽由 F = 4R/(1−R)² 决定。每增加 2π，透射曲线重复一次。",
                       fontsize=9, color=MUTED)

    def _build_ring_views(self):
        extent = 1 / math.sqrt(2)
        edges = np.linspace(-extent, extent, 502)
        coordinates = (edges[:-1] + edges[1:]) / 2
        self.radius = np.hypot(coordinates[:, None], coordinates[None, :])
        self.pixel_width = coordinates[1] - coordinates[0]
        self.illumination = np.exp(-2 * (self.radius / 0.69)**2)
        self.ring_axes = []
        self.ring_images = []
        self.ring_markers = []
        norm = PowerNorm(gamma=0.62, vmin=0, vmax=1)
        for left, title in ((0.517, "高斯照明"), (0.744, "均匀照明")):
            ax = self._axes([left, 0.507, 0.213, 0.310])
            self._title(ax, title, fontsize=12, color=INK, pad=10)
            ax.set(xticks=[], yticks=[])
            image = ax.imshow(np.zeros_like(self.radius), origin="lower", cmap=RING_COLORMAP, norm=norm,
                              extent=(-extent, extent, -extent, extent), interpolation="bilinear")
            marker, = ax.plot([], [], "o", ms=6, color="#c9f8ff", mfc="none", mew=1.0)
            self.ring_axes.append(ax)
            self.ring_images.append(image)
            self.ring_markers.append(marker)
        self.selection_text = self.fig.text(0.524, 0.449, "", fontsize=9, color=MUTED)

    def _build_train(self):
        self._fig_text(0.069, 0.402, "总光路", fontsize=15, weight="bold", color=INK)
        self._fig_text(0.304, 0.405, "青线为选中角度，橙色虚线多绕一圈，在焦平面汇到同一点。",
                       fontsize=9, color=MUTED)
        ax = self._axes([0.069, 0.162, 0.884, 0.219])
        e = MIRROR_SPACING
        ax.set(xlim=(-2.82 * e, 3.70 * e), ylim=(-2.95 * e, 3.50 * e), xticks=[], yticks=[])
        ax.axhline(0, color=BORDER, ls="--", lw=1.0)
        ax.axvspan(0, e, color="#edf5fd", zorder=0)
        for x in (0, e):
            ax.plot([x, x], [-2.28 * e, 2.72 * e], color=INK, lw=1.35, zorder=3)
        ax.plot([SCREEN_X, SCREEN_X], [-1.08 * e, 1.08 * e], color=INK, lw=2.7, zorder=3)

        # 镜片用透镜截面表示，追迹在其中心平面上采用理想薄透镜模型。
        section = np.linspace(-1, 1, 90)
        for x, height, half_thickness in (
                (DIVERGING_LENS_X, 1.35 * e, e * (0.052 + 0.112 * section**2)),
                (FOCUSING_LENS_X, 2.62 * e, e * (0.172 - 0.112 * section**2))):
            ax.fill_betweenx(section * height, x - half_thickness, x + half_thickness,
                             facecolor="#d7ecf8", edgecolor=BLUE, linewidth=1.0, alpha=0.83, zorder=2)
        labels = ((SOURCE_PLANE, "平行单色光"), (DIVERGING_LENS_X, "发散透镜"), (0, "镜 1"), (e, "镜 2"),
                  (FOCUSING_LENS_X, "会聚透镜"), (SCREEN_X, "焦平面屏"))
        for x, label in labels:
            self._axis_text(ax, x, 2.99 * e, label, ha="center", va="center", fontsize=9, color=INK)
        ax.annotate("", xy=(SCREEN_X, -1.77 * e), xytext=(FOCUSING_LENS_X, -1.77 * e),
                    arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
        self._axis_text(ax, (SCREEN_X + FOCUSING_LENS_X) / 2, -2.39 * e, "后焦距 f",
                        ha="center", fontsize=8, color=MUTED)

        self.context_rays = [ax.plot([], [], color=BLUE, alpha=0.27, lw=1.15)[0] for _ in range(5)]
        self.chosen_rays = [ax.plot([], [], color=TRANSMISSION, lw=1.85, zorder=4)[0] for _ in range(2)]
        self.extra_trip, = ax.plot([], [], color=RING_COLOR, lw=1.8, ls="--", zorder=5)
        self.direct_ray, = ax.plot([], [], color=ORANGE, alpha=0.65, ls=":", lw=1.25, zorder=4)
        self.screen_dots, = ax.plot([], [], "o", ms=5, color=RING_COLOR,
                                    markeredgecolor="white", markeredgewidth=0.55, zorder=6)
        self.incoming_arrow = self._new_arrow(ax, TRANSMISSION)
        self.roundtrip_arrow = self._new_arrow(ax, RING_COLOR)

    def _draw_train(self, cone, selected_slope, reflectance):
        stop = 3 if reflectance == 1 else None
        for line, fraction in zip(self.context_rays, (-0.91, -0.57, 0.0, 0.57, 0.91)):
            x, y = optical_train_ray(cone * fraction)
            line.set_data(x[:stop], y[:stop])
        for line, direction in zip(self.chosen_rays, (1, -1)):
            x, y = optical_train_ray(direction * selected_slope)
            line.set_data(x[:stop], y[:stop])

        tx, ty = optical_train_ray(selected_slope, round_trips=1)
        self.extra_trip.set_data(tx[3:], ty[3:])
        has_roundtrip = 0 < reflectance < 1
        self.extra_trip.set_visible(has_roundtrip)
        e = MIRROR_SPACING
        self.direct_ray.set_data([0, -0.70 * e], [ty[2], ty[2] + 0.70 * e * selected_slope])
        self.direct_ray.set_visible(reflectance > 0)
        focal_radius = FOCUSING_FOCAL_LENGTH * selected_slope
        self.screen_dots.set_data([SCREEN_X, SCREEN_X], [focal_radius, -focal_radius])
        self.screen_dots.set_visible(reflectance < 1)
        self._point_arrow(self.incoming_arrow, complex(tx[0] + 0.14 * e, ty[0]),
                          complex(tx[0] + 0.58 * e, ty[0]))
        start = complex(tx[3], ty[3])
        end = complex(tx[4], ty[4])
        self._point_arrow(self.roundtrip_arrow, start + 0.28 * (end - start), start + 0.58 * (end - start))
        self.roundtrip_arrow.set_visible(has_roundtrip)

    def _select_angle(self, event):
        if event.button == 1 and event.inaxes in self.ring_axes and event.xdata is not None and event.ydata is not None:
            self.selected = (event.xdata, event.ydata)
            self._refresh_selection()
            self.fig.canvas.draw_idle()

    def _refresh_selection(self):
        reflectance = self.r_slider.val
        cone = math.tan(math.radians(self.cone_slider.val))
        axis_phase = np.pi * (2 * RING_RESONANCE_ORDER + self.phase_slider.val)
        slope = math.hypot(*self.selected) * cone
        phase = axis_phase / math.sqrt(1 + slope**2)
        # 曲线以 2π 为周期，把所选角度的相位折回中间共振峰附近。
        phase_delta = math.remainder(phase, 2 * math.pi) / np.pi + 2
        transmittance = 0.0 if reflectance == 1 else airy_transmission(phase, airy_coefficient(reflectance))
        self.phase_cursor.set_xdata([phase_delta, phase_delta])
        self.phase_dot.set_data([phase_delta], [transmittance])
        for marker in self.ring_markers:
            marker.set_data([self.selected[0]], [self.selected[1]])
        self._message(self.selection_text, "点击圆环取样    θ = {angle:.2f}°    该角度透射率 {transmission:.2%}",
                      angle=math.degrees(math.atan(slope)), transmission=transmittance)
        self._draw_train(cone, slope, reflectance)

    def _refresh(self, _value):
        reflectance = self.r_slider.val
        if reflectance == 1:
            power = np.zeros_like(self.phase_grid)
        else:
            power = airy_transmission(self.phase_grid * np.pi, airy_coefficient(reflectance))
        self.current_curve.set_data(self.phase_grid, power)
        cone = math.tan(math.radians(self.cone_slider.val))
        axis_phase = np.pi * (2 * RING_RESONANCE_ORDER + self.phase_slider.val)
        pattern = screen_pattern(reflectance, axis_phase, cone, self.radius, self.pixel_width)
        self.ring_images[0].set_data(pattern * self.illumination)
        self.ring_images[1].set_data(pattern)
        self._refresh_selection()
        self.fig.canvas.draw_idle()


def main():
    configure_font()
    lesson = Lesson()
    plt.show()
    # 保留控件与计时器，直到关闭窗口。
    lesson.timer.stop()


if __name__ == "__main__":
    main()
