import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {

        AppGUI window = new AppGUI();
        window.frame.setSize(414, 400);
        window.frame.setResizable(false);
        window.frame.setVisible(true);

    }
}
