import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class AppGUI implements ActionListener {

    public  JFrame frame;
    public AppGUI(){
        frame = new JFrame();
        frame.getContentPane().setLayout(null);

        JButton jButtonAddItem = new JButton("Start Game of Life");
        jButtonAddItem.setBounds(170,300,200,50);
        jButtonAddItem.addActionListener(this);
        frame.add(jButtonAddItem);

        //TODO create input fields

        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }
    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        try {
            Runtime runTime = Runtime.getRuntime();

            String filePath = new File("").getAbsolutePath();

            //TODO change paths when release
            String executableFolder = filePath.substring(0, filePath.length() - 8) + "x64\\Debug\\";

            FileWriter myWriter = new FileWriter(executableFolder + "inputData.txt");

            //TODO get values from input fields
            myWriter.write("30" + "\n");
            myWriter.write("50" + "\n");
            myWriter.close();

            Process process = runTime.exec(executableFolder + "GameOfLife.exe");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
