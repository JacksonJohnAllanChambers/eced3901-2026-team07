/* UART DRIVER - TEAM 7 */

#ifndef UART_H
#define UART_H

// includes

#include <stdlib.h>
#include <avr/io.h>

// UART parameters

#define BAUD_RATE 103

// UART variables

volatile extern char uart_received;
volatile extern char uart_data;

// declare UART functions

void uart_init(uint16_t baud_rate);
void uart_putchar(char data);
void uart_putchar_async(char data);
void uart_putstring(char* s);
char uart_getchar(void);
void uart_putfloat(float f);

#endif