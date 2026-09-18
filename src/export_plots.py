"""
Exportador de gráficos clínicos de alta resolución para el README.md y documentación.
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
BG_CARD  = "#112236"
BG_MAIN  = "#0D1B2A"
BORDER   = "#1E3A5F"
TEXT_HI  = "#E8F4FD"
TEXT_LO  = "#7BA3C4"
ACCENT   = "#00C9A7"
RISK     = "#FF6B6B"
SAFE     = "#4ECDC4"

plt.rcParams.update({
    'figure.facecolor':  BG_CARD,
    'axes.facecolor':    BG_CARD,
    'axes.edgecolor':    BORDER,
    'axes.labelcolor':   TEXT_LO,
    'xtick.color':       TEXT_LO,
    'ytick.color':       TEXT_LO,
    'text.color':        TEXT_HI,
    'grid.color':        BORDER,
    'grid.alpha':        0.4,
    'font.size':         10
})

os.makedirs('img', exist_ok=True)
df = pd.read_csv('Data/processed/processed.csv')

# 1. Balance del Target
fig, ax = plt.subplots(figsize=(6, 4))
conteo = df['cardio'].value_counts().sort_index()
bars = ax.bar(['Sanos (0)', 'Enfermedad Cardiovascular (1)'], conteo.values, color=[SAFE, RISK], width=0.5)
for bar, val in zip(bars, conteo.values):
    pct = val / conteo.sum() * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
            f'{val:,}\n({pct:.1f}%)', ha='center', fontsize=10, color=TEXT_HI, fontweight='bold')
ax.set_ylabel('Pacientes')
ax.set_title('Balance de Variable Objetivo (cardio) - Cohorte Validada', color=TEXT_HI, pad=12)
ax.set_ylim(0, conteo.max() * 1.22)
plt.tight_layout()
fig.savefig('img/01_target_balance.png', dpi=200)
plt.close()

# 2. Violinplots de Variables Numéricas
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
vars_num = [('ap_hi', 'Presión Sistólica (mmHg)'),
            ('ap_lo', 'Presión Diastólica (mmHg)'),
            ('bmi', 'Índice de Masa Corporal (kg/m²)'),
            ('age_years', 'Edad (años)')]

palette_dict = {0: SAFE, 1: RISK, '0': SAFE, '1': RISK}

for ax, (col, titulo) in zip(axes.flatten(), vars_num):
    sns.violinplot(
        data=df, x='cardio', y=col, hue='cardio', ax=ax,
        palette=palette_dict, inner='box', linewidth=1.0, legend=False
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Sin Enfermedad (0)', 'Con Enfermedad (1)'])
    m0 = df[df['cardio']==0][col].mean()
    m1 = df[df['cardio']==1][col].mean()
    ax.axhline(m0, color=SAFE, linestyle='--', linewidth=1.2)
    ax.axhline(m1, color=RISK, linestyle='--', linewidth=1.2)
    ax.set_title(f'{titulo} | Media: {m0:.1f} vs {m1:.1f}', color=TEXT_HI, fontsize=11)
    ax.set_xlabel('')

plt.tight_layout()
fig.savefig('img/s2a_violinplots.png', dpi=200)
plt.close()

# 3. Categóricas vs Cardio
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
cols_cat = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'gender']
etiquetas = {
    'gender': {1: 'Mujer', 2: 'Hombre'},
    'cholesterol': {1: 'Normal', 2: 'Alto', 3: 'Muy alto'},
    'gluc': {1: 'Normal', 2: 'Alto', 3: 'Muy alto'},
    'smoke': {0: 'No fuma', 1: 'Fuma'},
    'alco': {0: 'No bebe', 1: 'Bebe'},
    'active': {0: 'Sedentario', 1: 'Activo'}
}
media_global = df['cardio'].mean() * 100

for ax, col in zip(axes.flatten(), cols_cat):
    tasa = df.groupby(col)['cardio'].mean() * 100
    etiq = [etiquetas[col].get(k, str(k)) for k in tasa.index]
    clrs = [RISK if v > media_global else SAFE for v in tasa.values]
    bars = ax.bar(etiq, tasa.values, color=clrs, width=0.55)
    ax.axhline(media_global, color='#FFD166', linestyle='--', linewidth=1.1, label=f'Media: {media_global:.1f}%')
    for bar, val in zip(bars, tasa.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
                f'{val:.1f}%', ha='center', fontsize=9, color=TEXT_HI, fontweight='bold')
    ax.set_title(col.capitalize(), color=TEXT_HI, fontsize=11)
    ax.set_ylabel('% Enfermedad (cardio=1)')
    ax.set_ylim(0, 90)
    ax.legend(fontsize=7.5, facecolor=BG_CARD, edgecolor=BORDER)

plt.tight_layout()
fig.savefig('img/s2b_categoricas_vs_cardio.png', dpi=200)
plt.close()

# 4. Matriz de Correlaciones
cols_corr = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo',
             'bmi', 'pulse_pressure', 'map', 'cholesterol', 'gluc',
             'smoke', 'alco', 'active', 'cardio']
corr = df[cols_corr].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
    center=0, vmin=-1, vmax=1, square=True, ax=ax,
    linewidths=0.4, linecolor=BG_MAIN, cbar_kws={'shrink': 0.8},
    annot_kws={'size': 8.5, 'color': TEXT_HI}
)
ax.set_title('Matriz de Correlación de Pearson entre Biomarcadores Clínicos', color=TEXT_HI, pad=14, fontsize=12)
plt.tight_layout()
fig.savefig('img/s4a_heatmap_correlaciones.png', dpi=200)
plt.close()

print("✓ Gráficos exportados exitosamente a la carpeta img/")
