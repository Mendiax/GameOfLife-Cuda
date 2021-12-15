import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.text.JTextComponent;
import javax.swing.text.NumberFormatter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.NumberFormat;

public class AppGUI implements ActionListener {

    public JFrame frame;

    JTextField jTextField_width;
    JTextField jTextField_height;

    String deathString = "000000000";
    String liveString = "000000000";

    public AppGUI() throws IOException {
        frame = new JFrame();
        frame.getContentPane().setLayout(null);
        frame.setTitle("Game of Life launcher");



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

        jTextField_width = new JTextField();
        jTextField_width.setBounds(10,40,200,30);
        frame.add(jTextField_width);

        JLabel horizontal = new JLabel();
        horizontal.setBounds(10,80,200,30);
        horizontal.setText("Number of cells (horizontal)");
        frame.add(horizontal);

        jTextField_height = new JTextField();
        jTextField_height.setBounds(10,110,200,30);
        frame.add(jTextField_height);

        JLabel survival = new JLabel();
        survival.setBounds(10,150,200,30);
        survival.setText("Survival rules");
        frame.add(survival);


        ItemListener itemListener = itemEvent -> {
            JToggleButton temp = (JToggleButton) itemEvent.getItem();
            int state = itemEvent.getStateChange();
            if (state == ItemEvent.SELECTED) {
                if(temp.getName().equals("L")){
                    char[] myNameChars = liveString.toCharArray();
                    myNameChars[Integer.parseInt(temp.getText())] = '1';
                    liveString = String.valueOf(myNameChars);
                }
                else {
                    char[] myNameChars = deathString.toCharArray();
                    myNameChars[Integer.parseInt(temp.getText())] = '1';
                    deathString = String.valueOf(myNameChars);
                }
            }
            else {
                if(temp.getName().equals("L")){
                    char[] myNameChars = liveString.toCharArray();
                    myNameChars[Integer.parseInt(temp.getText())] = '0';
                    liveString = String.valueOf(myNameChars);
                }
                else {
                    char[] myNameChars = deathString.toCharArray();
                    myNameChars[Integer.parseInt(temp.getText())] = '0';
                    deathString = String.valueOf(myNameChars);
                }
            }
        };

        for ( int i = 0; i < 9; i++){
            JToggleButton jToggleButton = new JToggleButton();
            jToggleButton.setBounds(10 + i*42,180,42,42);
            jToggleButton.addItemListener(itemListener);
            jToggleButton.setName("L");
            jToggleButton.setText(String.valueOf(i));
            frame.add(jToggleButton);
        }

        JLabel death = new JLabel();
        death.setBounds(10,220,200,30);
        death.setText("Death rules");
        frame.add(death);

        for ( int i = 0; i < 9; i++){
            JToggleButton jToggleButton = new JToggleButton();
            jToggleButton.setBounds(10 + i*42,250,42,42);
            jToggleButton.setText(String.valueOf(i));
            jToggleButton.setName("D");
            jToggleButton.addItemListener(itemListener);
            frame.add(jToggleButton);
        }

        JButton jButtonAddItem = new JButton("Start simulation");
        jButtonAddItem.setBounds(10,300,184,50);
        jButtonAddItem.addActionListener(this);
        frame.add(jButtonAddItem);

        ActionListener info = itemEvent -> {
            System.out.println("ASD");
            JOptionPane.showMessageDialog(frame,
                    "Controls:\n" +
                    "         LMB - change state of selected cell\n" +
                    "         RMP - start/stop simulation\n" +
                    "         Arrow keys - move board\n" +
                    "         Scroll - zoom in/out\n" +
                    "Creators:\n" +
                    "         Jakub Quasner\n" +
                    "         Bartłomiej Dzikowski\n" +
                    "         Ksawery Możdżyński\n" +
                    "Survival Rules:\n" +
                    "         Number of neighbors cells that is required for each\n" +
                    "         living cell to survive in next generation\n" +
                    "Death Rules:\n" +
                    "         Number of neighbors cells that is required for each\n" +
                    "         death cell to become alive in next generation\n",
                    "Info",
                    JOptionPane.INFORMATION_MESSAGE);
        };

        JButton jButtonInfo = new JButton("Info");
        jButtonInfo.setBounds(204,300,184,50);
        jButtonInfo.addActionListener(info);
        frame.add(jButtonInfo);


        String filePath = new File("").getAbsolutePath();
        System.out.println(filePath);
        BufferedImage myPicture = ImageIO.read(new File( "gol.png"));
        JLabel picLabel = new JLabel(new ImageIcon(myPicture));
        picLabel.setBounds(200,10,200,150);
        frame.add(picLabel);

        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }
    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        try {


            String executableFolder = new File("").getAbsolutePath();

            FileWriter myWriter = new FileWriter(executableFolder + "\\inputData.txt");

            int w = 0;
            int h = 0;
            try {
                w = Integer.parseInt(jTextField_width.getText());
                h = Integer.parseInt(jTextField_height.getText());
                if (w < 0 || h < 0){
                    JOptionPane.showMessageDialog(frame,
                            "Number of cells should be a positive number",
                            "NegativeNumberException",
                            JOptionPane.ERROR_MESSAGE);
                    myWriter.close();
                    return;
                }
            }catch (NumberFormatException e){
                JOptionPane.showMessageDialog(frame,
                        "Number of cells is should be a number",
                        "NumberFormatException",
                        JOptionPane.ERROR_MESSAGE);
                myWriter.close();
                return;
            }
            myWriter.write(jTextField_width.getText() + "\n");
            myWriter.write(jTextField_height.getText() + "\n");
            myWriter.write(liveString + "\n");
            myWriter.write(deathString + "\n");
            myWriter.close();

            Runtime runTime = Runtime.getRuntime();
            Process process = runTime.exec(executableFolder + "\\GameOfLife.exe");
            System.exit(0);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
