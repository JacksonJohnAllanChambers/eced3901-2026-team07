/* LIGHTING MODULE - TEAM 7*/

#ifndef LIGHTING_H
#define LIGHTING_H

// includes

#include <avr/io.h>
#include <stdlib.h>

// define fsk parameters

#define TOP_1K 250
#define TOP_3K 84
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
#define DANGER_THRESHOLD 60
#define SAFETY_THRESHOLD 120

// declare lighting functions

void lighting_init(void);
void fsk_handler(void);
void ultrasonic_tick(void);
void ultrasonic_update(void);
float ultrasonic_get_distance(void);
void LED_update(void);


#endif
