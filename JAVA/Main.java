import java.util.Random;
import java.util.Scanner;  // Import the Scanner class

public class Main {
    public static void main(String[] args) {
        Scanner myObj = new Scanner(System.in);  // Create a Scanner object
        boolean guessedIt = false;
        Random random = new Random();
        int randomNumber = random.nextInt(1, 101);
        int attempts = 0;

        while(!guessedIt)
        {
            System.out.println("Skus si tipnut ake to je cislo:");
            int guess = myObj.nextInt();  // Read user input
            attempts++;

            if(guess < 1 || guess > 100 )
                System.out.println("U fucked it!");
            else
            {
                if(guess == randomNumber)
                {
                    System.out.println("Spravne!");
                    System.out.println("Pocet pokusov: " + attempts);
                    guessedIt = true;
                }
                else
                {
                    if(guess > randomNumber)
                        System.out.println("Skus nizsie!");
                    else
                        System.out.println("Skus vyssie!");    
                }
            }
        }
        
        System.out.println("Tajne cislo bolo: " + randomNumber);
    }
}