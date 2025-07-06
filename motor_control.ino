// Pin Definitions
#define ENA 6       // Motor speed control (PWM)
#define IN1 9       // Motor direction control
#define IN2 10      
#define POT A0      // Potentiometer
#define RED_LED 7   // Failure LED
#define GREEN_LED 8 // Normal LED

bool isMotorRunning = true; // Track motor state

void setup() {
    pinMode(ENA, OUTPUT);
    pinMode(IN1, OUTPUT);
    pinMode(IN2, OUTPUT);
    pinMode(RED_LED, OUTPUT);
    pinMode(GREEN_LED, OUTPUT);
    pinMode(POT, INPUT);

    Serial.begin(9600);
    startMotor();  // Start motor initially
}

void loop() {
    int potValue = analogRead(POT); // Read potentiometer value (0-1023)
    int motorSpeed = map(potValue, 0, 1023, 0, 255); // Convert to PWM (0-255)
    int rpm = map(potValue, 0, 1023, 0, 3000); // Simulated RPM

    Serial.print("RPM: ");
    Serial.println(rpm);
    
    analogWrite(ENA, motorSpeed); // Adjust motor speed dynamically

    // Control motor and LEDs based on RPM threshold
    if (rpm < 300) {  
        if (isMotorRunning) {  // Only stop if running
            stopMotor();
            isMotorRunning = false;
        }
    } else {
        if (!isMotorRunning) { // Only start if stopped
            startMotor();
            isMotorRunning = true;
        }
    }

    // Check for Serial Commands (Optional)
    if (Serial.available()) {
        String command = Serial.readStringUntil('\n');
        command.trim();

        if (command == "FAILURE") {
            stopMotor();
            isMotorRunning = false;
        } else if (command == "SAFE") {
            startMotor();
            isMotorRunning = true;
        }
    }

    delay(500); // Reduce delay for more responsive updates
}

// Function to Start Motor & Set LED
void startMotor() {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, 255);  // Ensure motor starts at full speed

    digitalWrite(GREEN_LED, HIGH);
    digitalWrite(RED_LED, LOW);

    Serial.println("Motor Running - SAFE");
}

// Function to Stop Motor & Set LED
void stopMotor() {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, 0); // Ensure motor stops

    digitalWrite(GREEN_LED, LOW);
    digitalWrite(RED_LED, HIGH);

    Serial.println("Motor Stopped - FAILURE");
}
