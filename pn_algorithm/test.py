import matplotlib.pyplot as plt

eq = r'$\dot{\chi}_{\rm cmd} = \frac{\gamma}{t_{go}} + N\,\dot{\chi}_{LOS}$'
plt.figure(figsize=(4,1))
plt.text(0.5, 0.5, eq, fontsize=20, ha='center', va='center')
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('chi_cmd_equation.png', dpi=300, transparent=True)
