import javax.swing.*;
import java.awt.event.ActionEvent;

public class Main {
    public static void main(String[] args) {

        AppGUI window = new AppGUI();
        window.frame.setSize(400, 400);
        window.frame.setResizable(false);
        window.frame.setVisible(true);

    }
}
