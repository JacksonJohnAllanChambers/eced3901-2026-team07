#include "schedule.h"
#include "cargo.h"
#include "uart.h"
#include "lighting.h"
#include "rescue.h"
#include <avr/interrupt.h>

volatile char uart_received = 0;
volatile char uart_data = 0;

ISR(TIMER0_COMPA_vect){
    scheduler_tick();
    ultrasonic_tick();
    fsk_handler();
    virtual_pwm_tick();
}

ISR(USART_RX_vect){
    uart_data = UDR0;
    uart_received = 1;
}

int main(void){
    // initialize variables
    char cargo_pickup = 0, rescue = 0, byte = 0;
    // pull up
    PORTC |= (1<<PINC2);

    // initialize i/o pins and registers
    lighting_init(); 
    cargo_init();
    uart_init(BAUD_RATE);
    rescue_init();

    // enable global interrupts
    sei();

    while(1){
        // continuously poll ultrasonic sensor
        ultrasonic_update();
        LED_update();
       
        if(uart_received){
            uart_received = 0;
            byte = uart_data;
            switch(byte){
                case 0:
                    cargo_pickup = 0;
                    uart_putchar('I');
                    break;
                case 1:
                    cargo_pickup = 1;
                    uart_putchar('Y');
                    break;
                case 2:
                    cargo_pickup = 2;
                    uart_putchar('N');
                    break;
                case 3:
                    rescue = 0;
                    uart_putchar('I');
                    break;
                case 4:
                    rescue = 1;
                    uart_putchar('R');
                    break;
                default:
                    break;
            }    
        }

        cargo_update(cargo_pickup);
        rescue_boat(rescue);
        cargo_pickup = 0;
        if(!(PINC & (1 << PINC2))){
            cargo_pickup = 1;
            rescue = 1;
        }
    }
}