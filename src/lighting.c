/* LIGHTING MODULE IMPLEMENTATION*/

#include "lighting.h"

// Counter variables
static volatile uint8_t next_bit = 0, i = 0;
static volatile uint16_t delay_count = 0;
static volatile uint32_t count = 0, ucount = 0;

// other variables
static uint16_t current = 0;
static float last_distance = 0;
static char checking = 0, red = 0, green = 0, yellow = 0;

// message array
static const uint8_t sequence[MESSAGE_LEN] = {
    0,1,0,0,0,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,0,0,1,0
};

void lighting_init(void){

    /* Scheduling Timer Setup */

    // Enable CTC mode
    TCCR0A = (1<<WGM01);
    // Set COMPA value for 20us timer
    OCR0A = 39; 
    // Initialize timer register
    TCNT0 = 0;
    // Enable COMPA interrupts
    TIMSK0 |= (1<<OCIE0A);
    // Set prescaler to 8
    TCCR0B |= (1<<CS01);

    /* FSK Setup */
    
    // Enable OC2B as output
    DDRD |= (1<<FSK_OUTPUT_PIN);
    // Enable pull-up on electromagnet control pin
    PORTD |= (1<<CONTROL_PIN);
    // Enable non-inverting Fast PWM, OCR2A TOP, prescaler 64
    TCCR2A |= (1<<WGM20) | (1<<WGM21) | (1<<COM2B1);
    TCCR2B |= (1<<WGM22) | (1<<CS22);

    /* Ultrasonic Setup */

    // Enable pinout
    DDRB |= (1<<TRIG_PIN) | (1<<LED_RED_PIN) | (1<<LED_GREEN_PIN) | (1<<LED_YELLOW_PIN);

    // enable global interrupts
    sei();

}

void fsk_handler(void){
    // Increment count
    count++;

    // Handle bit timing
    if(count >= 454){ // ~9ms (bit period) has passed
        next_bit = 1; // move to next bit
        count = 0; // reset count
    }

    // Handle message logic
    if(i < MESSAGE_LEN){ // message has not been fully sent yet
        DDRD |= (1<<FSK_OUTPUT_PIN); // ensure output is on

        // Find frequency of next bit
        if(next_bit == 1){
            if(sequence[i] == 1){
                OCR2A = TOP_3K; 
                OCR2B = TOP_3K/2; // 50% duty cycle
            }
            else{
                OCR2A = TOP_1K;
                OCR2B = TOP_1K/2; // 50% duty cycle
            }
            i++; // Move along message 
            next_bit = 0; // Bit has been sent, wait for next bit
        }
    }
    else if (i == MESSAGE_LEN){ // last bit being sent
        if(next_bit == 1){
			i++;
			next_bit = 0;
		}
    }
    else { // message has been fully sent, small delay until next message
        // Turn off output
        DDRD &= ~(1<<FSK_OUTPUT_PIN);

        // Send FSK message once per second
        if(++delay_count > 15000){
            i = 0; // loop back around and start a new message
            delay_count = 0; // restart message timer
        }
    }

}

void ultrasonic_tick(void){
    // update counter
    ucount++; 
    // 60ms timing schedule
    if(ucount > 3000) ucount = 0; 
    // send trig pin a 20us pulse
    if(ucount == 1) PORTB |= (1<<TRIG_PIN);
    if(ucount == 2) PORTB &= ~(1<<TRIG_PIN);
}

void ultrasonic_update(void){
    // if echo pin goes high
    if((PINB & (1<<ECHO_PIN)) && checking == 0){
        // save current timestamp
        current = ucount;
        // start checking for change in echo pin
        checking = 1;
    }

    // wait for echo pin to go low
    if(!(PINB & (1<<ECHO_PIN)) && checking == 1){
        checking = 0;
        last_distance = (ucount - current)*0.343; // convert to cm
    }

}

float ultrasonic_get_distance(void){
    return last_distance;
}

void LED_update(void){

    // Update LED state
    if(last_distance <= DANGER_THRESHOLD){
        red++; // 
        yellow = 0;
        green = 0;
        if(red > 3){
            PORTB |= (1<<LED_RED_PIN);
			PORTB &= ~((1<<LED_YELLOW_PIN) | (1<<LED_GREEN_PIN));
			red = 0;
        }
    }
    else if (last_distance <= SAFETY_THRESHOLD){
        yellow++;
        red = 0;
        green = 0;
        if(yellow > 3){
            PORTB |= (1<<LED_YELLOW_PIN);
			PORTB &= ~((1<<LED_RED_PIN) | (1<<LED_GREEN_PIN));
			yellow = 0;
        }
    }
    else {
        green++;
        red = 0;
        yellow = 0;
        if(green > 3){
			PORTB |= (1<<LED_GREEN_PIN);
			PORTB &= ~((1<<LED_YELLOW_PIN) | (1<<LED_RED_PIN));
			green = 0;
		}
    }

}