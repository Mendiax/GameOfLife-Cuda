import javax.swing.*;
import javax.swing.text.NumberFormatter;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.NumberFormat;

public class AppGUI implements ActionListener {

    public JFrame frame;

    JFormattedTextField jTextField_width;
    JFormattedTextField jTextField_height;

    public AppGUI() {
        frame = new JFrame();
        frame.getContentPane().setLayout(null);
        frame.setTitle("Game of Life launcher");

        JButton jButtonAddItem = new JButton("Start Game of Life");
        jButtonAddItem.setBounds(10,300,200,50);
        jButtonAddItem.addActionListener(this);
        frame.add(jButtonAddItem);

        NumberFormat format = NumberFormat.getInstance();
        NumberFormatter formatter = new NumberFormatter(format);
        formatter.setValueClass(Integer.class);
        formatter.setMinimum(0);
        formatter.setMaximum(Integer.MAX_VALUE);
        formatter.setAllowsInvalid(false);
        formatter.setCommitsOnValidEdit(false);

        JLabel vertical = new JLabel();
        vertical.setBounds(10,10,200,30);
        vertical.setText("Number of cells (vertical)");
        frame.add(vertical);

        jTextField_width = new JFormattedTextField(formatter);
        jTextField_width.setBounds(10,40,200,30);
        //jTextField_width.setText("10");
        frame.add(jTextField_width);

        JLabel horizontal = new JLabel();
        horizontal.setBounds(10,80,200,30);
        horizontal.setText("Number of cells (horizontal)");
        frame.add(horizontal);

        jTextField_height = new JFormattedTextField(formatter);
        jTextField_height.setBounds(10,110,200,30);
        //jTextField_height.setText("10");
        frame.add(jTextField_height);

        JLabel survival = new JLabel();
        survival.setBounds(10,150,200,30);
        survival.setText("Survival rules (TODO)");
        frame.add(survival);

        JTextField jTextField_survival = new JTextField();
        jTextField_survival.setBounds(10,180,200,30);
        frame.add(jTextField_survival);

        JLabel death = new JLabel();
        death.setBounds(10,220,200,30);
        death.setText("Death rules (TODO)");
        frame.add(death);

        JTextField jTextField_death = new JTextField();
        jTextField_death.setBounds(10,250,200,30);
        frame.add(jTextField_death);

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

            myWriter.write(jTextField_width.getText() + "\n");
            myWriter.write(jTextField_height.getText() + "\n");
            myWriter.close();

            Process process = runTime.exec(executableFolder + "GameOfLife.exe");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
