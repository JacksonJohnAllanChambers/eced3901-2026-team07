/* LIGHTING MODULE - TEAM 7*/

#ifndef LIGHTING_MODULE
#define LIGHTING_MODULE

// define fsk parameters

#define TOP_1K 84
#define TOP_3K 250
#define MESSAGE_LEN 24
#define FSK_OUTPUT_PIN PIND3

// define safety light parameters

#define ECHO_PIN PINB0
#define TRIG_PIN PINB1
#define LED_GREEN_PIN PINB2
#define LED_YELLOW_PIN PINB3
#define LED_RED_PIN PINB4
#define CONTROL_PIN PIND7
#define MAX_ECHO_TIMEOUT 2500

// define counter variables

extern volatile uint32_t count, ucount, ucount2;
extern volatile uint16_t delay_count;

// declare lighting functions

void lighting_init(void);

void fsk_handler(uint8_t *sequence, uint8_t size);








#endif
