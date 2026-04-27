import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import random


class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 11)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def init_model():
    model = SimpleNeuralNetwork()

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    model.apply(init_weights)
    return model


model = init_model()
print("New neural network created (not trained)")


class TrainingData:
    def __init__(self):
        self.images = []
        self.labels = []

    def add_sample(self, image_tensor, label):
        self.images.append(image_tensor.clone())
        self.labels.append(label)
        label_name = "empty" if label == 10 else str(label)
        print(f"Added example: {label_name}")

    def get_loader(self, batch_size=32):
        if len(self.images) == 0:
            return None
        images_tensor = torch.cat(self.images, dim=0)
        labels_tensor = torch.tensor(self.labels)
        dataset = TensorDataset(images_tensor, labels_tensor)
        return DataLoader(dataset, batch_size=min(batch_size, len(self.images)), shuffle=True)

    def clear(self):
        self.images = []
        self.labels = []

    def __len__(self):
        return len(self.images)


class DigitDrawApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.model.train()

        self.training_data = TrainingData()
        self.training_history = []

        self.root.title("Neural Network Training - Learn Digits")
        self.root.geometry("1300x900")
        self.root.configure(bg='#2c3e50')

        self.last_x = None
        self.last_y = None
        self.drawing = False

        self.image_size = 280
        self.mnist_size = 28
        self.image = Image.new('L', (self.image_size, self.image_size), 'black')
        self.draw = ImageDraw.Draw(self.image)

        self.current_prediction = None
        self.current_confidence = 0
        self.current_image = None
        self.last_recognition_result = None

        self.create_widgets()
        self.update_info_labels()

    def create_widgets(self):
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        left_panel = tk.Frame(main_container, bg='#2c3e50')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(left_panel, text="DRAW A DIGIT",
                 font=('Arial', 18, 'bold'), bg='#2c3e50', fg='white').pack(pady=10)

        self.canvas = tk.Canvas(left_panel, width=self.image_size, height=self.image_size,
                                bg='black', cursor='cross', highlightthickness=2, highlightbackground='#3498db')
        self.canvas.pack(pady=10)

        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.start_paint)
        self.canvas.bind('<ButtonRelease-1>', self.stop_paint)

        btn_frame = tk.Frame(left_panel, bg='#2c3e50')
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text="CLEAR", command=self.clear_canvas,
                  font=('Arial', 12, 'bold'), bg='#e74c3c', fg='white', padx=30, pady=10).pack(side=tk.LEFT, padx=10)

        tk.Button(btn_frame, text="RECOGNIZE", command=self.recognize_digit,
                  font=('Arial', 12, 'bold'), bg='#3498db', fg='white', padx=30, pady=10).pack(side=tk.LEFT, padx=10)

        right_panel = tk.Frame(main_container, bg='#2c3e50')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))

        tk.Label(right_panel, text="RECOGNITION RESULT",
                 font=('Arial', 18, 'bold'), bg='#2c3e50', fg='white').pack(pady=10)

        result_frame = tk.Frame(right_panel, bg='#34495e', relief=tk.RAISED, bd=3)
        result_frame.pack(fill=tk.X, pady=10, padx=10)

        self.result_label = tk.Label(result_frame, text="???", font=('Arial', 88, 'bold'),
                                     bg='#34495e', fg='#3498db')
        self.result_label.pack(pady=30)

        confidence_frame = tk.Frame(right_panel, bg='#34495e', relief=tk.RAISED, bd=3)
        confidence_frame.pack(fill=tk.X, pady=10, padx=10)

        tk.Label(confidence_frame, text="CONFIDENCE", font=('Arial', 12, 'bold'),
                 bg='#34495e', fg='white').pack(pady=(10, 0))

        self.confidence_label = tk.Label(confidence_frame, text="0%", font=('Arial', 28, 'bold'),
                                         bg='#34495e', fg='#2ecc71')
        self.confidence_label.pack(pady=10)

        self.confidence_bar = ttk.Progressbar(confidence_frame, length=250, mode='determinate')
        self.confidence_bar.pack(pady=10)

        training_frame = tk.Frame(right_panel, bg='#2c3e50')
        training_frame.pack(fill=tk.X, pady=10)

        self.correct_btn = tk.Button(training_frame, text="CORRECT (Add to training)",
                                     command=self.add_correct_example,
                                     font=('Arial', 12, 'bold'), bg='#2ecc71', fg='white',
                                     padx=20, pady=12, state=tk.NORMAL)
        self.correct_btn.pack(fill=tk.X, pady=5)

        self.wrong_btn = tk.Button(training_frame, text="WRONG (Fix error)",
                                   command=self.add_corrected_example,
                                   font=('Arial', 12, 'bold'), bg='#e74c3c', fg='white',
                                   padx=20, pady=12, state=tk.DISABLED)
        self.wrong_btn.pack(fill=tk.X, pady=5)

        tk.Button(training_frame, text="TRAIN EMPTY", command=self.add_empty_example,
                  font=('Arial', 12), bg='#95a5a6', fg='white', padx=20, pady=12).pack(fill=tk.X, pady=5)

        self.train_btn = tk.Button(training_frame, text="TRAIN NETWORK",
                                   command=self.train_network,
                                   font=('Arial', 14, 'bold'), bg='#9b59b6', fg='white',
                                   padx=20, pady=12)
        self.train_btn.pack(fill=tk.X, pady=10)

        tk.Button(training_frame, text="TEST MODEL", command=self.test_random,
                  font=('Arial', 12), bg='#1abc9c', fg='white', padx=20, pady=10).pack(fill=tk.X, pady=5)

        graph_frame = tk.Frame(right_panel, bg='#2c3e50')
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(5, 3.5))
        self.fig.patch.set_facecolor('#2c3e50')
        self.ax.set_facecolor('#34495e')
        self.canvas_plot = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.probabilities = np.zeros(11)
        self.update_plot()

        bottom_panel = tk.Frame(self.root, bg='#34495e', relief=tk.RAISED, bd=2)
        bottom_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(0, 20))

        stats_frame = tk.Frame(bottom_panel, bg='#34495e')
        stats_frame.pack(fill=tk.X, padx=10, pady=10)

        self.examples_label = tk.Label(stats_frame, text="EXAMPLES: 0", font=('Arial', 11, 'bold'),
                                       bg='#34495e', fg='#3498db')
        self.examples_label.pack(side=tk.LEFT, padx=20)

        self.train_count_label = tk.Label(stats_frame, text="TRAININGS: 0", font=('Arial', 11, 'bold'),
                                          bg='#34495e', fg='#2ecc71')
        self.train_count_label.pack(side=tk.LEFT, padx=20)

        self.accuracy_label = tk.Label(stats_frame, text="ACCURACY: 0%", font=('Arial', 11, 'bold'),
                                       bg='#34495e', fg='#f39c12')
        self.accuracy_label.pack(side=tk.LEFT, padx=20)

        tk.Button(bottom_panel, text="FULL RESET", command=self.reset_everything,
                  font=('Arial', 10, 'bold'), bg='#e74c3c', fg='white', padx=15, pady=5).pack(side=tk.RIGHT, padx=10,
                                                                                              pady=5)

    def start_paint(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def paint(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=20, fill='white', capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y], fill='white', width=20)
            self.last_x = x
            self.last_y = y

    def stop_paint(self, event):
        self.drawing = False

    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (self.image_size, self.image_size), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="???")
        self.confidence_label.config(text="0%")
        self.confidence_bar['value'] = 0
        self.probabilities = np.zeros(11)
        self.update_plot()
        self.wrong_btn.config(state=tk.DISABLED)
        self.last_recognition_result = None

    def preprocess_image(self):
        img_resized = self.image.resize((self.mnist_size, self.mnist_size), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = 255 - img_array
        img_array = img_array / 255.0
        img_array = (img_array - 0.5) / 0.5
        img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
        return img_tensor

    def is_empty(self):
        img_array = np.array(self.image)
        white_pixels = np.sum(img_array > 200)
        return white_pixels < 100

    def recognize_digit(self):
        if self.is_empty():
            self.result_label.config(text="EMPTY")
            self.confidence_label.config(text="100%")
            self.confidence_bar['value'] = 100
            self.probabilities = np.zeros(11)
            self.probabilities[10] = 1.0
            self.update_plot()
            self.current_prediction = 10
            self.current_confidence = 100
            self.wrong_btn.config(state=tk.DISABLED)
            return

        img_tensor = self.preprocess_image()
        self.current_image = img_tensor.clone()

        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted = torch.max(output, 1)[1].item()
            confidence = probabilities[0][predicted].item() * 100

        self.current_prediction = predicted
        self.current_confidence = confidence

        display_text = "EMPTY" if predicted == 10 else str(predicted)
        self.result_label.config(text=display_text)
        self.confidence_label.config(text=f"{confidence:.1f}%")
        self.confidence_bar['value'] = confidence
        self.probabilities = probabilities[0].cpu().numpy()
        self.update_plot()

        self.wrong_btn.config(state=tk.NORMAL)
        self.last_recognition_result = predicted

        self.result_label.config(fg='#2ecc71')
        self.root.after(500, lambda: self.result_label.config(fg='#3498db'))

    def add_correct_example(self):
        if self.is_empty():
            messagebox.showwarning("Warning", "Draw a digit first!")
            return

        correct = simpledialog.askinteger("Training",
                                          "What digit is drawn? (0-9)",
                                          minvalue=0, maxvalue=9, parent=self.root)

        if correct is not None:
            self.training_data.add_sample(self.current_image, correct)
            self.update_info_labels()

            messagebox.showinfo("Success",
                                f"Digit {correct} added to training!\n"
                                f"Total examples: {len(self.training_data)}\n\n"
                                f"Press 'TRAIN NETWORK' to train!")

            self.clear_canvas()

    def add_corrected_example(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Recognize a digit first!")
            return

        correct = simpledialog.askinteger("Fix Error",
                                          f"Model showed: {self.current_prediction if self.current_prediction != 10 else 'EMPTY'}\n"
                                          f"What is the correct digit? (0-9)",
                                          minvalue=0, maxvalue=9, parent=self.root)

        if correct is not None:
            self.training_data.add_sample(self.current_image, correct)
            self.update_info_labels()

            messagebox.showinfo("Fixed",
                                f"Model will remember this is digit {correct}\n"
                                f"Total examples: {len(self.training_data)}\n\n"
                                f"Press 'TRAIN NETWORK' to fix the error!")

            self.clear_canvas()
            self.wrong_btn.config(state=tk.DISABLED)

    def add_empty_example(self):
        empty_image = Image.new('L', (self.mnist_size, self.mnist_size), 'black')
        empty_array = np.array(empty_image, dtype=np.float32) / 255.0
        empty_array = (empty_array - 0.5) / 0.5
        empty_tensor = torch.tensor(empty_array).unsqueeze(0).unsqueeze(0)

        self.training_data.add_sample(empty_tensor, 10)
        self.update_info_labels()

        messagebox.showinfo("Done",
                            "Empty example added!\nModel will learn to recognize empty canvas!")

    def train_network(self):
        if len(self.training_data) < 2:
            messagebox.showwarning("Not enough data!",
                                   "Need at least 2 examples to train!\n\n"
                                   "1. Draw a digit\n"
                                   "2. Press 'CORRECT' or 'WRONG'\n"
                                   "3. Repeat several times")
            return

        progress = tk.Toplevel(self.root)
        progress.title("Training Neural Network")
        progress.geometry("400x150")
        tk.Label(progress, text=f"Training on {len(self.training_data)} examples...",
                 font=('Arial', 12)).pack(pady=20)
        prog_bar = ttk.Progressbar(progress, length=300, mode='indeterminate')
        prog_bar.pack(pady=10)
        prog_bar.start()
        self.root.update()

        self.model.train()
        train_loader = self.training_data.get_loader()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        num_epochs = max(20, min(100, len(self.training_data) * 5))

        print(f"\n{'=' * 60}")
        print(f"TRAINING STARTED on {len(self.training_data)} EXAMPLES")
        print(f"{'=' * 60}")

        final_accuracy = 0

        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total if total > 0 else 0
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - Accuracy: {accuracy:.1f}%")
            final_accuracy = accuracy

        self.model.eval()
        with torch.no_grad():
            all_correct = 0
            all_total = 0
            for images, labels in train_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                all_total += labels.size(0)
                all_correct += (predicted == labels).sum().item()

        final_accuracy = 100 * all_correct / all_total if all_total > 0 else 0

        self.training_history.append({
            'examples': len(self.training_data),
            'accuracy': final_accuracy,
            'epochs': num_epochs
        })

        prog_bar.stop()
        progress.destroy()

        messagebox.showinfo("TRAINING COMPLETE!",
                            f"NEURAL NETWORK TRAINED!\n\n"
                            f"Results:\n"
                            f"   Examples: {len(self.training_data)}\n"
                            f"   Epochs: {num_epochs}\n"
                            f"   Accuracy: {final_accuracy:.1f}%\n\n"
                            f"Model now knows these digits!\n"
                            f"Draw them again to test!")

        self.update_info_labels()

        torch.save(self.model.state_dict(), 'my_trained_model.pth')
        print("Model saved to 'my_trained_model.pth'")

        self.clear_canvas()

    def test_random(self):
        if len(self.training_data) == 0:
            messagebox.showwarning("Warning", "Train the model first!")
            return

        test_window = tk.Toplevel(self.root)
        test_window.title("Model Testing")
        test_window.geometry("500x400")
        test_window.configure(bg='#2c3e50')

        tk.Label(test_window, text="MODEL TEST", font=('Arial', 16, 'bold'),
                 bg='#2c3e50', fg='white').pack(pady=10)

        examples_frame = tk.Frame(test_window, bg='#2c3e50')
        examples_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas_frame = tk.Frame(examples_frame, bg='#2c3e50')
        canvas_frame.pack()

        test_canvas = tk.Canvas(canvas_frame, width=140, height=140, bg='black')
        test_canvas.pack(pady=10)

        result_label = tk.Label(examples_frame, text="", font=('Arial', 24, 'bold'),
                                bg='#2c3e50', fg='white')
        result_label.pack(pady=10)

        idx = [0]
        images_list = self.training_data.images
        labels_list = self.training_data.labels

        def show_example():
            if idx[0] >= len(images_list):
                test_window.destroy()
                return

            img = images_list[idx[0]].squeeze().cpu().numpy()
            img = (img + 1) / 2
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img, mode='L')
            img = img.resize((140, 140), Image.Resampling.NEAREST)

            test_canvas.delete("all")
            img_tk = ImageTk.PhotoImage(img)
            test_canvas.create_image(70, 70, image=img_tk)
            test_canvas.image = img_tk

            with torch.no_grad():
                output = self.model(images_list[idx[0]].unsqueeze(0))
                predicted = torch.max(output, 1)[1].item()
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence = probs[0][predicted].item() * 100

            true_label = labels_list[idx[0]]
            true_name = "empty" if true_label == 10 else str(true_label)
            pred_name = "empty" if predicted == 10 else str(predicted)

            if predicted == true_label:
                result_label.config(text=f"CORRECT! Digit: {pred_name}", fg='#2ecc71')
            else:
                result_label.config(text=f"WRONG! Should be: {true_name}, Model: {pred_name}", fg='#e74c3c')

        def next_example():
            idx[0] += 1
            show_example()

        tk.Button(examples_frame, text="NEXT EXAMPLE", command=next_example,
                  font=('Arial', 12), bg='#3498db', fg='white', padx=20, pady=10).pack(pady=10)

        def close_test():
            test_window.destroy()

        tk.Button(examples_frame, text="CLOSE", command=close_test,
                  font=('Arial', 12), bg='#e74c3c', fg='white', padx=20, pady=10).pack(pady=5)

        show_example()

    def reset_everything(self):
        if messagebox.askyesno("FULL RESET",
                               "Are you sure?\n\n"
                               "Model will be recreated\n"
                               "All examples will be deleted\n"
                               "All statistics will be reset\n\n"
                               "Continue?"):
            global model
            self.model = init_model()
            model = self.model
            self.training_data.clear()
            self.training_history = []
            self.update_info_labels()
            self.clear_canvas()

            messagebox.showinfo("Done!", "Model reset! You can start training again.")

    def update_info_labels(self):
        self.examples_label.config(text=f"EXAMPLES: {len(self.training_data)}")
        self.train_count_label.config(text=f"TRAININGS: {len(self.training_history)}")

        if self.training_history:
            avg_acc = sum(h['accuracy'] for h in self.training_history) / len(self.training_history)
            self.accuracy_label.config(text=f"ACCURACY: {avg_acc:.1f}%")
        else:
            self.accuracy_label.config(text="ACCURACY: 0%")

    def update_plot(self):
        self.ax.clear()
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'E']

        bars = self.ax.bar(range(11), self.probabilities, color='#3498db', alpha=0.7, edgecolor='white')

        if len(self.probabilities) > 0:
            max_idx = np.argmax(self.probabilities)
            if self.probabilities[max_idx] > 0:
                bars[max_idx].set_color('#e74c3c')
                bars[max_idx].set_alpha(0.9)

        self.ax.set_xlabel('What neural network sees', color='white', fontsize=11)
        self.ax.set_ylabel('Probability', color='white', fontsize=11)
        self.ax.set_xticks(range(11))
        self.ax.set_xticklabels(classes, color='white')
        self.ax.set_ylim(0, 1)
        self.ax.tick_params(colors='white')
        self.ax.set_facecolor('#34495e')
        self.ax.grid(True, alpha=0.3, axis='y')

        for i, (bar, prob) in enumerate(zip(bars, self.probabilities)):
            if prob > 0.05:
                self.ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                             f'{prob:.0%}', ha='center', va='bottom', fontsize=9, color='white')

        self.fig.tight_layout()
        self.canvas_plot.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDrawApp(root, model)

    instruction = """
    HOW TO TRAIN THE NEURAL NETWORK:

    1. TRAINING:
       Draw a digit -> press "RECOGNIZE"
       If CORRECT: press "CORRECT" -> enter the digit
       If WRONG: press "WRONG" -> enter the correct digit

    2. START TRAINING:
       Press "TRAIN NETWORK" (after 3-5 examples)

    3. TESTING:
       Press "TEST MODEL" - see how well it learned

    TIP: Start with digits 0,1,2,3 - 2-3 examples each
    """

    tk.Label(root, text=instruction, font=('Arial', 9), bg='#2c3e50',
             fg='#f39c12', justify=tk.LEFT).pack(side=tk.BOTTOM, pady=10)

    root.mainloop()
