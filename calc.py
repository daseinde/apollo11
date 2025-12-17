import math
import csv
from data import *  # Импортируем все переменные из data.py

# Вспомогательные расчеты
# Массовые расходы для каждой ступени
# Массовый расход = разница между полной и пустой массой ступени/время работы двигателя этой ступени.
dm1 = (massOfStage1Full - massOfStage1Empty) / stageEngineDuration[0]  # кг/с
dm2 = (massOfStage2Full - massOfStage2Empty) / stageEngineDuration[1]
dm3 = (massOfStage3Full - massOfStage3Empty) / stageEngineDuration[2]

# расчет силы тяги для каждой ступени: удельные имп двиг * ускор своб падения * масс расход
F1 = IspOfSideEngine * g0_kerbin * dm1  # 1-я ступень
F2 = IspOfCentralEngine * g0_kerbin * dm2  # 2-я ступень
F3 = IspOfManeurEngine * g0_kerbin * dm3  # 3-я ступень

# Время окончания работы ступеней
t_stage1_end = stagePeriods[0]  # 150 с
t_stage2_end = stagePeriods[0] + stagePeriods[1]  # 380 с
t_stage3_end = stagePeriods[0] + stagePeriods[1] + stagePeriods[2]  # 600 с

# Начальные условия
t = 0  # время, с
h = 0  # высота, м
v = 0  # скорость, м/с
dt = 1  # шаг по времени, с

# Масса без топлива (после сброса всех ступеней)
dry_mass = Me

# Текущая масса (начинаем с полной)
current_mass = Mf

# Списки масс ступеней
stage1_mass = massOfStage1Full
stage2_mass = massOfStage2Full
stage3_mass = massOfStage3Full

# Создаем список для хранения данных
data = []

# Моделируем до указанного периода
while t <= period:
    # 1. Определяем текущую ступень и силу тяги
    if t < t_stage1_end:
        F_thrust = F1
        # Уменьшаем массу первой ступени
        elapsed_time = t
        # гарантия того, что масса первой ступени не станет меньше, чем ее пустая масса
        stage1_mass = max(massOfStage1Empty, massOfStage1Full - dm1 * elapsed_time)
        # Общая масса: масса без топлива + оставшиеся ступени
        current_mass = dry_mass + stage2_mass + stage3_mass + stage1_mass
    elif t < t_stage2_end:
        F_thrust = F2
        t_stage2 = t - t_stage1_end
        # Уменьшаем массу второй ступени
        stage2_mass = max(massOfStage2Empty, massOfStage2Full - dm2 * t_stage2)
        # Первая ступень уже сброшена
        current_mass = dry_mass + stage3_mass + stage2_mass
    elif t < t_stage3_end:
        F_thrust = F3
        t_stage3 = t - t_stage2_end
        # Уменьшаем массу третьей ступени
        stage3_mass = max(massOfStage3Empty, massOfStage3Full - dm3 * t_stage3)
        # Первая и вторая ступени сброшены
        current_mass = dry_mass + stage3_mass
    else:
        F_thrust = 0
        # Все ступени сброшены
        current_mass = dry_mass

    # 2. Сила гравитации (зависит от высоты) - ДЛЯ KERBIN!
    g = g0_kerbin * (RadiusOfKerbin ** 2) / ((RadiusOfKerbin + h) ** 2)
    F_gravity = current_mass * g

    # 3. Атмосферное давление (барометрическая формула для Kerbin)
    # Высота шкалы для Kerbin: ~5000 м (5 км)
    scale_height = 5000  # м, для Kerbin
    
    # БАРОВАЯ ФОРМУЛА: P = P0 * exp(-h/H)
    if h <= 70000:  # В атмосфере Kerbin
        P = P0_kerbin * math.exp(-h / scale_height)
    else:
        P = 0  # За атмосферой

    # 4. Плотность среды (для Kerbin) - используем уравнение состояния идеального газа
    if h <= 70000:  # В атмосфере Kerbin
        # ρ = P * M / (R * T)
        rho = P * M_kerbin / (R_gas * T_kerbin)
    else:
        rho = 0  # Вакуум

    # 5. Сила лобового сопротивления (только в атмосфере, всегда направлена против скорости)
    if rho > 0:
        # В KSP коэффициент сопротивления обычно около 0.2 для ракет
        # Используем абсолютное значение скорости для направления силы
        F_drag = 0.5 * Cf * S * rho * v * abs(v)
    else:
        F_drag = 0

    # 6. Ускорение (второй закон Ньютона)
    # Суммарная сила: тяга - сопротивление - гравитация
    # Направление: при движении вверх v>0, при падении v<0
    if current_mass > 0:
        # Для упрощения считаем движение только вертикально вверх без наклона
        a = (F_thrust - F_drag - F_gravity) / current_mass
    else:
        a = 0

    # Форматируем значения для лучшей читаемости
    time_val = int(t)
    thrust_val = round(F_thrust, 1)
    mass_val = round(current_mass, 1)
    gravity_val = round(F_gravity, 1)
    pressure_val = round(P, 2)
    density_val = round(rho, 6)
    drag_val = round(F_drag, 1)
    accel_val = round(a, 4)
    velocity_val = round(v, 2)
    altitude_val = round(h, 2)

    # Сохраняем данные
    data.append([
        time_val, thrust_val, mass_val, gravity_val, 
        pressure_val, density_val, drag_val, 
        accel_val, velocity_val, altitude_val
    ])

    # Интегрирование (метод Эйлера) для следующего шага
    v = v + a * dt
    h = h + v * dt
    
    # Увеличиваем время
    t += dt

# Сохраняем в CSV файл
headers = [
    "Time (s)", "Thrust (N)", "Mass (kg)", "Gravity (N)", 
    "Pressure (Pa)", "Density (kg/m3)", "Drag (N)", 
    "Acceleration (m/s2)", "Velocity (m/s)", "Altitude (m)"
]

with open('apollo11_simulation_ksp.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(data)
