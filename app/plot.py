import matplotlib.pyplot as plt
import base64
from io import BytesIO

def plot_sequence(sequence):
    fig, ax = plt.subplots()
    sequence = sequence.T  # shape: [features][time]
    for dim in sequence:
        ax.plot(dim)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
