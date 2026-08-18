# Machine Learning — Spring 2025

**Course:** IML4042 — Machine Learning (Spring 2025)

This repository contains my solutions, reports, and learning materials for the **Machine Learning** course taught by **Dr. Amirhossein Nikoofard** in Spring 2025.

Everything is organized into three main areas:

- 📚 **Lectures** — the slide decks covered in class (Deep Learning, MLP, CNN, RNN, Transformers, GAN, TinyML, and more)
- ✍️ **Homework** — five assignments (HW1–HW5), each with the original problem statement, my typed **LaTeX report** (PDF), the Python/PyTorch/JAX code, and interactive **Jupyter notebooks**
- 🎓 **Final Exam** — the final exam paper and my solutions

---

## 🗂 Repository Structure

```
Machine_Learning_Spring-2025_Dr.Amirhossein-Nikoofard/
│
├── lectures/                  # Lecture slides (PDF)
│
├── hw/
│   ├── hw1/                   # Assignment 1 — ML foundations & backpropagation
│   ├── hw2/                   # Assignment 2 — CNNs & autoencoders
│   ├── hw3/                   # Assignment 3 — RNNs & LSTMs
│   ├── hw4/                   # Assignment 4 — Dimensionality reduction & GANs
│   └── hw5/                   # Assignment 5 — Transformers & MiniGPT
│
├── Final Exam/                # Final exam paper + suggested solutions
│
└── docs/
    ├── lectures/README.md     # Lecture-by-lecture index
    ├── hw1/README.md          # HW1 overview
    ├── hw2/README.md          # HW2 overview
    ├── hw3/README.md          # HW3 overview
    ├── hw4/README.md          # HW4 overview
    ├── hw5/README.md          # HW5 overview
    └── final-exam/README.md   # Final exam overview
```

## 📖 What's Inside

| Folder | Topic | Highlights |
|--------|-------|------------|
| [`lectures/`](lectures/) | Deep Learning fundamentals → advanced architectures | MLP, CNN, RNN, Transformers, GANs, TinyML |
| [`hw/hw1/`](hw/hw1/) | NNs, backpropagation, ADAM | "Habitable Zone" classification, manual backprop |
| [`hw/hw2/`](hw/hw2/) | CNNs, bottleneck blocks, autoencoders | Learnable conv masks, Dice-loss YOLO, contractive autoencoder |
| [`hw/hw3/`](hw/hw3/) | RNNs, LSTMs, sequence modeling | CIFG-LSTM from scratch, vanila RNN parity, Robust BiLSTM |
| [`hw/hw4/`](hw/hw4/) | Dimensionality reduction & generative models | PCA/Isomap on CIFAR-10, Conditional DCGAN on BreastMNIST |
| [`hw/hw5/`](hw/hw5/) | Transformers & autoregressive language models | MiniGPT built in JAX — tokenization → training → sampling |
| [`Final Exam/`](Final Exam/) | Comprehensive final exam | IML4042 final paper |

> For a detailed view of any section, open the corresponding README under [`docs/`](docs/).

## 🛠 Technologies Used

- **Python** — NumPy, scikit-learn
- **PyTorch** — torch.nn, torchvision, torchinfo
- **JAX** — Flax-based Transformer experiments (HW5)
- **Keras** — CIFAR-10 data loading (HW4)
- **Jupyter** — interactive notebooks for every assignment
- **LaTeX** — professional typesetting of all reports

## 📝 Note on Course Materials

- `hw/hw4/view.php` is actually a ZIP archive (a download artifact); it contains the original HW4 report PDF for the Conditional DCGAN (Q2).
- `hw/hw5/صورت سوالات و راهنمای حل تمرین.pdf` is the original HW5 problem statement and solution guide (in Persian).
- Some homework folders contain multiple compiled `main*.pdf` files — they are earlier/alternative builds of the same LaTeX report.

## 🙏 Acknowledgments

Special thanks to **Dr. Amirhossein Nikoofard** for the course, the TAs for their guidance and feedback, and to the instructors of the open-source teaching materials (e.g., the Transformer/JAX labs used in HW5, originally by the Snapp! / Digikala / Tapsi / Quera / Divar ML teams).