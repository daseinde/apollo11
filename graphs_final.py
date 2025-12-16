import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import os

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Создаем директорию для графиков если ее нет
os.makedirs('ksp_vs_saturnv_plots', exist_ok=True)

# Загружаем данные из KSP
ksp_data = pd.read_csv('apollo11_simulation_ksp.csv')

# Параметры для реального Saturn V (на основе технических характеристик)
# используется для сравнения с результатами моделирования

class SaturnVSimulation:
    def __init__(self):
        # Более точные параметры Saturn V
        self.params = {
            'total_mass_start': 2.95e6,      # кг
            'first_stage_dry': 0.131e6,      # сухая масса 1й ступени
            'second_stage_start': 0.496e6,   # масса в начале 2й ступени
            'second_stage_dry': 0.039e6,     # сухая масса 2й ступени
            'thrust_first_sl': 33.4e6,       # Н на уровне моря
            'thrust_first_vac': 40.0e6,      # Н в вакууме (увеличивается!)
            'thrust_second': 5.0e6,          # Н (5x J-2)
            'burn_time_first': 168,          # с (фактическое время)
            'burn_time_second': 360,         # с
            'diameter': 10.1,                # м
            'cd': 0.5,                       # коэффициент сопротивления
            'g0': 9.81,                      # м/с²
            'R_earth': 6371e3                # м
        }
        
        self.params['A'] = np.pi * (self.params['diameter']/2)**2
        
        # Параметры атмосферы
        self.p0 = 101325
        self.rho0 = 1.225
        self.H = 8500

    # Упрощенная модель атмосферы
    def atmosphere_model(self, h):
        if h < 0:
            h = 0
        p = self.p0 * np.exp(-h/self.H)
        rho = self.rho0 * np.exp(-h/self.H)
        return p, rho

    # рассчитываем изменение тяги в зависимости от высоты.
    def thrust_variation(self, h, thrust_sl, thrust_vac):
        scale_height = 30000  # высота, на которой тяга близка к вакуумной
        if h < scale_height:
            return thrust_sl + (thrust_vac - thrust_sl) * (h / scale_height)
        return thrust_vac

    # выполняем расчет полета ракеты и сохраняем данные
    def simulate(self, t_max=528, dt=1):
        time = np.arange(0, t_max + dt, dt)
        n = len(time)
        
        # Инициализация массивов (для отслеживания изменений каждого параметра)
        thrust = np.zeros(n)
        mass = np.zeros(n)
        gravity_force = np.zeros(n)
        pressure = np.zeros(n)
        density = np.zeros(n)
        drag = np.zeros(n)
        acceleration = np.zeros(n)
        velocity = np.zeros(n)
        altitude = np.zeros(n)
        
        # Начальные условия
        mass[0] = self.params['total_mass_start']
        altitude[0] = 0
        velocity[0] = 0
        
        # Массовые расходы
        # 1-я ступень: расход 13,100 кг/с
        m_dot_first = (self.params['total_mass_start'] - 
                      (self.params['first_stage_dry'] + self.params['second_stage_start'])) / self.params['burn_time_first']
        
        # 2-я ступень: расход 1,270 кг/с
        m_dot_second = (self.params['second_stage_start'] - 
                       self.params['second_stage_dry']) / self.params['burn_time_second']
 # метод Эйлера (i-1) нужен для получения некоторых данных из прошлых расчетов
        for i in range(1, n):
            t = time[i]
            
            # Определяем текущую ступень
            if t <= self.params['burn_time_first']:
                # 1-я ступень
                thrust[i] = self.thrust_variation(
                    altitude[i-1], 
                    self.params['thrust_first_sl'],
                    self.params['thrust_first_vac']
                )
                mass[i] = mass[i-1] - m_dot_first * dt
            else:
                # 2-я ступень (после отделения 1й)
                if i == int(self.params['burn_time_first']/dt) + 1:
                    # Момент отделения 1й ступени
                    mass[i] = self.params['second_stage_start']
                thrust[i] = self.params['thrust_second']
                if mass[i-1] > self.params['second_stage_dry']:
                    mass[i] = mass[i-1] - m_dot_second * dt
                else:
                    mass[i] = self.params['second_stage_dry']
                    thrust[i] = 0
            
            # Атмосферные условия
            p, rho = self.atmosphere_model(altitude[i-1])
            pressure[i] = p  # сохраняем давление
            density[i] = rho
            
            # Гравитация
            g = self.params['g0'] * (self.params['R_earth'] / (self.params['R_earth'] + altitude[i-1]))**2
            gravity_force[i] = mass[i] * g
            
            # Сопротивление
            if altitude[i-1] < 100000:
                drag[i] = 0.5 * rho * velocity[i-1]**2 * self.params['cd'] * self.params['A']
            else:
                drag[i] = 0
            
            # Ускорение
            net_force = thrust[i] - drag[i] - gravity_force[i]
            acceleration[i] = net_force / mass[i] if mass[i] > 0 else 0
            
            # Интегрирование
            velocity[i] = velocity[i-1] + acceleration[i] * dt
            altitude[i] = altitude[i-1] + velocity[i] * dt
        
        # Создаем DataFrame
        saturnv_df = pd.DataFrame({
            'Время (с)': time,
            'Тяга (Н)': thrust,
            'Масса (кг)': mass,
            'Гравитация (Н)': gravity_force,
            'Давление (Па)': pressure,
            'Плотность (кг/м³)': density,
            'Сопротивление (Н)': drag,
            'Ускорение (м/с²)': acceleration,
            'Скорость (м/с)': velocity,
            'Высота (м)': altitude,
            'Скорость (км/ч)': velocity * 3.6
        })
        
        return saturnv_df


# Создаем симуляцию Saturn V
saturnv_sim = SaturnVSimulation()
saturnv_data = saturnv_sim.simulate(t_max=380, dt=1)
print(f"Создана симуляция Saturn V с {len(saturnv_data)} записями")

# Для сравнения обрежем данные KSP до 380 секунд если нужно
if len(ksp_data) > 380:
    ksp_data = ksp_data[ksp_data['Time (s)'] <= 380].copy()

# Переименуем колонки в данных KSP для совместимости
ksp_data = ksp_data.rename(columns={
    'Time (s)': 'Время (с)',
    'Thrust (N)': 'Тяга (Н)',
    'Mass (kg)': 'Масса (кг)',
    'Gravity (N)': 'Гравитация (Н)',
    'Pressure (Pa)': 'Давление (Па)',
    'Density (kg/m3)': 'Плотность (кг/м³)',
    'Drag (N)': 'Сопротивление (Н)',
    'Acceleration (m/s2)': 'Ускорение (м/с²)',
    'Velocity (m/s)': 'Скорость (м/с)',
    'Altitude (m)': 'Высота (м)'
})

# Создаем все графики
figures = []

# 1. Зависимость высоты от времени
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(ksp_data['Время (с)'], ksp_data['Высота (м)'] / 1000, 
         'b-', linewidth=2, label='Симуляция KSP', alpha=0.8)
ax1.plot(saturnv_data['Время (с)'], saturnv_data['Высота (м)'] / 1000, 
         'r--', linewidth=2, label='Saturn V (расчет)', alpha=0.8)
ax1.set_xlabel('Время (с)', fontsize=12)
ax1.set_ylabel('Высота (км)', fontsize=12)
ax1.set_title('Высота vs Время', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 380)
figures.append(('altitude_vs_time', fig1))

# 2. Зависимость массы от времени
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(ksp_data['Время (с)'], ksp_data['Масса (кг)'] / 1000, 
         'b-', linewidth=2, label='Симуляция KSP', alpha=0.8)
ax2.plot(saturnv_data['Время (с)'], saturnv_data['Масса (кг)'] / 1000, 
         'r--', linewidth=2, label='Saturn V (расчет)', alpha=0.8)
ax2.set_xlabel('Время (с)', fontsize=12)
ax2.set_ylabel('Масса (тонны)', fontsize=12)
ax2.set_title('Масса vs Время', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 380)
figures.append(('mass_vs_time', fig2))

# 3. Зависимость давления от высоты
fig3, ax3 = plt.subplots(figsize=(12, 6))
# Для KSP
ax3.plot(ksp_data['Высота (м)'] / 1000, ksp_data['Давление (Па)'], 
        'b-', linewidth=2, label='Симуляция KSP', alpha=0.6)
# Для Saturn V
ax3.plot(saturnv_data['Высота (м)'] / 1000, saturnv_data['Давление (Па)'], 
        'r--', linewidth=2, label='Saturn V (расчет)', alpha=0.8)
ax3.set_xlabel('Высота (км)', fontsize=12)
ax3.set_ylabel('Давление (Па)', fontsize=12)
ax3.set_title('Давление vs Высота', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 50)  # Ограничиваем высоту для лучшей видимости
ax3.set_ylim(0, 110000)  # Устанавливаем разумные пределы для давления
figures.append(('pressure_vs_altitude', fig3))

# 4. Зависимость силы сопротивления от высоты
fig4, ax4 = plt.subplots(figsize=(12, 6))
# Для KSP
ax4.plot(ksp_data['Высота (м)'] / 1000, ksp_data['Сопротивление (Н)'] / 1000, 
        'b-', linewidth=2, label='Симуляция KSP', alpha=0.8)
# Для Saturn V
ax4.plot(saturnv_data['Высота (м)'] / 1000, saturnv_data['Сопротивление (Н)'] / 1000, 
        'r--', linewidth=2, label='Saturn V (расчет)', alpha=0.8)
ax4.set_xlabel('Высота (км)', fontsize=12)
ax4.set_ylabel('Сила сопротивления (кН)', fontsize=12)
ax4.set_title('Сила сопротивления vs Высота', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 100)
ax4.set_ylim(0, max(ksp_data['Сопротивление (Н)'].max(), saturnv_data['Сопротивление (Н)'].max()) / 1000 * 1.1)
figures.append(('drag_vs_altitude', fig4))

# 5. Зависимость скорости от времени
fig5, ax5 = plt.subplots(figsize=(12, 6))
ax5.plot(ksp_data['Время (с)'], ksp_data['Скорость (м/с)'], 
         'b-', linewidth=2, label='Симуляция KSP', alpha=0.8)
ax5.plot(saturnv_data['Время (с)'], saturnv_data['Скорость (м/с)'], 
         'r--', linewidth=2, label='Saturn V (расчет)', alpha=0.8)
ax5.set_xlabel('Время (с)', fontsize=12)
ax5.set_ylabel('Скорость (м/с)', fontsize=12)
ax5.set_title('Скорость vs Время', fontsize=14, fontweight='bold')
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, 380)
figures.append(('velocity_vs_time', fig5))

# 6. Зависимость силы гравитации от времени
fig6, ax6 = plt.subplots(figsize=(12, 6))
ax6.plot(ksp_data['Время (с)'], ksp_data['Гравитация (Н)'] / 1e6, 
         'b-', linewidth=2, label='Симуляция KSP', alpha=0.8)
ax6.plot(saturnv_data['Время (с)'], saturnv_data['Гравитация (Н)'] / 1e6, 
         'r--', linewidth=2, label='Saturn V (расчет)', alpha=0.8)
ax6.set_xlabel('Время (с)', fontsize=12)
ax6.set_ylabel('Сила гравитации (МН)', fontsize=12)
ax6.set_title('Сила гравитации vs Время', fontsize=14, fontweight='bold')
ax6.legend(fontsize=11)
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, 380)
figures.append(('gravity_vs_time', fig6))

# 7. Зависимость плотности воздуха от высоты
fig7, ax7 = plt.subplots(figsize=(12, 6))
# Для KSP
ax7.plot(ksp_data['Высота (м)'] / 1000, ksp_data['Плотность (кг/м³)'], 
        'b-', linewidth=2, label='Симуляция KSP', alpha=0.6)
# Для Saturn V
ax7.plot(saturnv_data['Высота (м)'] / 1000, saturnv_data['Плотность (кг/м³)'], 
        'r--', linewidth=2, label='Saturn V (расчет)', alpha=0.8)
ax7.set_xlabel('Высота (км)', fontsize=12)
ax7.set_ylabel('Плотность (кг/м³)', fontsize=12)
ax7.set_title('Плотность атмосферы vs Высота', fontsize=14, fontweight='bold')
ax7.legend(fontsize=11)
ax7.grid(True, alpha=0.3)
ax7.set_xlim(0, 50)  # Ограничиваем высоту для лучшей видимости
ax7.set_ylim(0, 1.3)  # Устанавливаем разумные пределы для плотности
figures.append(('density_vs_altitude', fig7))

# 8. Зависимость силы тяги от времени
fig8, ax8 = plt.subplots(figsize=(12, 6))
ax8.plot(ksp_data['Время (с)'], ksp_data['Тяга (Н)'] / 1e6, 
         'b-', linewidth=2, label='Симуляция KSP', alpha=0.8)
ax8.plot(saturnv_data['Время (с)'], saturnv_data['Тяга (Н)'] / 1e6, 
         'r--', linewidth=2, label='Saturn V (расчет)', alpha=0.8)
ax8.set_xlabel('Время (с)', fontsize=12)
ax8.set_ylabel('Тяга (МН)', fontsize=12)
ax8.set_title('Тяга vs Время', fontsize=14, fontweight='bold')
ax8.legend(fontsize=11)
ax8.grid(True, alpha=0.3)
ax8.set_xlim(0, 380)
figures.append(('thrust_vs_time', fig8))

# Сохраняем все графики
for name, fig in figures:
    filepath = f'ksp_vs_saturnv_plots/{name}.png'
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Сохранен: {filepath}")
    plt.close(fig)

# Создаем сводную таблицу сравнения ключевых параметров
print("СВОДКА СРАВНЕНИЯ KSP И SATURN V")

# Ключевые точки сравнения
comparison_points = [60, 120, 150, 200, 300, 380]

print("\nКлючевые параметры на разных этапах полета:")
print("-" * 100)
print(f"{'Время (с)':<12} {'Параметр':<22} {'KSP':<20} {'Saturn V':<20} {'Отношение (KSP/SatV)':<20}")
print("-" * 100)

for t in comparison_points:
    if t in ksp_data['Время (с)'].values and t in saturnv_data['Время (с)'].values:
        ksp_row = ksp_data[ksp_data['Время (с)'] == t].iloc[0]
        satv_row = saturnv_data[saturnv_data['Время (с)'] == t].iloc[0]
        
        # Высота
        print(f"{t:<12} {'Высота (км)':<22} {ksp_row['Высота (м)']/1000:<20.1f} {satv_row['Высота (м)']/1000:<20.1f} {(ksp_row['Высота (м)']/satv_row['Высота (м)']):<20.2f}")
        
        # Скорость
        print(f"{'':<12} {'Скорость (м/с)':<22} {ksp_row['Скорость (м/с)']:<20.1f} {satv_row['Скорость (м/с)']:<20.1f} {(ksp_row['Скорость (м/с)']/satv_row['Скорость (м/с)']):<20.2f}")
        
        # Масса
        print(f"{'':<12} {'Масса (тонны)':<22} {ksp_row['Масса (кг)']/1000:<20.1f} {satv_row['Масса (кг)']/1000:<20.1f} {(ksp_row['Масса (кг)']/satv_row['Масса (кг)']):<20.2f}")
        
        print("-" * 100)


print(f"Всего создано {len(figures)} графиков в папке 'ksp_vs_saturnv_plots/'")

# Выводим путь к файлам
print("\nФайлы с графиками доступны по следующим путям:")
for name, _ in figures:
    print(f"  ksp_vs_saturnv_plots/{name}.png")