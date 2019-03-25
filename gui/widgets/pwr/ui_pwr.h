/********************************************************************************
** Form generated from reading UI file 'pwr.ui'
**
** Created by: Qt User Interface Compiler version 5.9.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PWR_H
#define UI_PWR_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLCDNumber>
#include <QtWidgets/QLabel>
#include <QtWidgets/QWidget>
#include "xqwt_multi_dial.h"

QT_BEGIN_NAMESPACE

class Ui_pwr
{
public:
    QLCDNumber *totalCurrent;
    QLCDNumber *averageVoltage;
    QLCDNumber *averageFrequency;
    XqwtMultiDial *currentDial;
    QWidget *gridLayoutWidget;
    QGridLayout *switchGrid;
    XqwtMultiDial *voltageDial;
    XqwtMultiDial *frequencyDial;
    QLabel *ampsLabel;
    QLabel *ampsLabel_2;
    QLabel *ampsLabel_3;

    void setupUi(QWidget *pwr)
    {
        if (pwr->objectName().isEmpty())
            pwr->setObjectName(QStringLiteral("pwr"));
        pwr->resize(759, 468);
        QPalette palette;
        QBrush brush(QColor(32, 31, 31, 255));
        brush.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Base, brush);
        QBrush brush1(QColor(30, 29, 29, 255));
        brush1.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Window, brush1);
        palette.setBrush(QPalette::Inactive, QPalette::Base, brush);
        palette.setBrush(QPalette::Inactive, QPalette::Window, brush1);
        palette.setBrush(QPalette::Disabled, QPalette::Base, brush1);
        palette.setBrush(QPalette::Disabled, QPalette::Window, brush1);
        pwr->setPalette(palette);
        totalCurrent = new QLCDNumber(pwr);
        totalCurrent->setObjectName(QStringLiteral("totalCurrent"));
        totalCurrent->setGeometry(QRect(100, 160, 81, 61));
        QPalette palette1;
        QBrush brush2(QColor(9, 175, 175, 255));
        brush2.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::WindowText, brush2);
        QBrush brush3(QColor(64, 63, 62, 0));
        brush3.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::Button, brush3);
        QBrush brush4(QColor(4, 3, 3, 0));
        brush4.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::Light, brush4);
        QBrush brush5(QColor(61, 60, 60, 0));
        brush5.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::Midlight, brush5);
        QBrush brush6(QColor(28, 25, 16, 0));
        brush6.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::Dark, brush6);
        QBrush brush7(QColor(212, 210, 207, 255));
        brush7.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::Text, brush7);
        QBrush brush8(QColor(232, 230, 227, 255));
        brush8.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::ButtonText, brush8);
        palette1.setBrush(QPalette::Active, QPalette::Base, brush);
        QBrush brush9(QColor(30, 29, 29, 0));
        brush9.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::Window, brush9);
        palette1.setBrush(QPalette::Inactive, QPalette::WindowText, brush2);
        palette1.setBrush(QPalette::Inactive, QPalette::Button, brush3);
        palette1.setBrush(QPalette::Inactive, QPalette::Light, brush4);
        palette1.setBrush(QPalette::Inactive, QPalette::Midlight, brush5);
        palette1.setBrush(QPalette::Inactive, QPalette::Dark, brush6);
        palette1.setBrush(QPalette::Inactive, QPalette::Text, brush7);
        palette1.setBrush(QPalette::Inactive, QPalette::ButtonText, brush8);
        palette1.setBrush(QPalette::Inactive, QPalette::Base, brush);
        palette1.setBrush(QPalette::Inactive, QPalette::Window, brush9);
        palette1.setBrush(QPalette::Disabled, QPalette::WindowText, brush6);
        palette1.setBrush(QPalette::Disabled, QPalette::Button, brush3);
        palette1.setBrush(QPalette::Disabled, QPalette::Light, brush4);
        palette1.setBrush(QPalette::Disabled, QPalette::Midlight, brush5);
        palette1.setBrush(QPalette::Disabled, QPalette::Dark, brush6);
        palette1.setBrush(QPalette::Disabled, QPalette::Text, brush6);
        palette1.setBrush(QPalette::Disabled, QPalette::ButtonText, brush6);
        palette1.setBrush(QPalette::Disabled, QPalette::Base, brush9);
        palette1.setBrush(QPalette::Disabled, QPalette::Window, brush9);
        totalCurrent->setPalette(palette1);
        totalCurrent->setSmallDecimalPoint(true);
        totalCurrent->setDigitCount(4);
        totalCurrent->setSegmentStyle(QLCDNumber::Filled);
        totalCurrent->setProperty("value", QVariant(8));
        totalCurrent->setProperty("intValue", QVariant(8));
        averageVoltage = new QLCDNumber(pwr);
        averageVoltage->setObjectName(QStringLiteral("averageVoltage"));
        averageVoltage->setGeometry(QRect(310, 160, 101, 61));
        QPalette palette2;
        QBrush brush10(QColor(24, 224, 224, 255));
        brush10.setStyle(Qt::SolidPattern);
        palette2.setBrush(QPalette::Active, QPalette::WindowText, brush10);
        palette2.setBrush(QPalette::Active, QPalette::Button, brush3);
        palette2.setBrush(QPalette::Active, QPalette::Light, brush4);
        palette2.setBrush(QPalette::Active, QPalette::Midlight, brush5);
        palette2.setBrush(QPalette::Active, QPalette::Dark, brush6);
        palette2.setBrush(QPalette::Active, QPalette::Text, brush7);
        palette2.setBrush(QPalette::Active, QPalette::ButtonText, brush8);
        palette2.setBrush(QPalette::Active, QPalette::Base, brush);
        palette2.setBrush(QPalette::Active, QPalette::Window, brush9);
        palette2.setBrush(QPalette::Inactive, QPalette::WindowText, brush10);
        palette2.setBrush(QPalette::Inactive, QPalette::Button, brush3);
        palette2.setBrush(QPalette::Inactive, QPalette::Light, brush4);
        palette2.setBrush(QPalette::Inactive, QPalette::Midlight, brush5);
        palette2.setBrush(QPalette::Inactive, QPalette::Dark, brush6);
        palette2.setBrush(QPalette::Inactive, QPalette::Text, brush7);
        palette2.setBrush(QPalette::Inactive, QPalette::ButtonText, brush8);
        palette2.setBrush(QPalette::Inactive, QPalette::Base, brush);
        palette2.setBrush(QPalette::Inactive, QPalette::Window, brush9);
        palette2.setBrush(QPalette::Disabled, QPalette::WindowText, brush6);
        palette2.setBrush(QPalette::Disabled, QPalette::Button, brush3);
        palette2.setBrush(QPalette::Disabled, QPalette::Light, brush4);
        palette2.setBrush(QPalette::Disabled, QPalette::Midlight, brush5);
        palette2.setBrush(QPalette::Disabled, QPalette::Dark, brush6);
        palette2.setBrush(QPalette::Disabled, QPalette::Text, brush6);
        palette2.setBrush(QPalette::Disabled, QPalette::ButtonText, brush6);
        palette2.setBrush(QPalette::Disabled, QPalette::Base, brush9);
        palette2.setBrush(QPalette::Disabled, QPalette::Window, brush9);
        averageVoltage->setPalette(palette2);
        averageVoltage->setSmallDecimalPoint(true);
        averageVoltage->setDigitCount(5);
        averageVoltage->setSegmentStyle(QLCDNumber::Filled);
        averageVoltage->setProperty("value", QVariant(119.8));
        averageFrequency = new QLCDNumber(pwr);
        averageFrequency->setObjectName(QStringLiteral("averageFrequency"));
        averageFrequency->setGeometry(QRect(570, 160, 81, 61));
        QPalette palette3;
        palette3.setBrush(QPalette::Active, QPalette::WindowText, brush10);
        palette3.setBrush(QPalette::Active, QPalette::Button, brush3);
        palette3.setBrush(QPalette::Active, QPalette::Light, brush4);
        palette3.setBrush(QPalette::Active, QPalette::Midlight, brush5);
        palette3.setBrush(QPalette::Active, QPalette::Dark, brush6);
        palette3.setBrush(QPalette::Active, QPalette::Text, brush7);
        palette3.setBrush(QPalette::Active, QPalette::ButtonText, brush8);
        palette3.setBrush(QPalette::Active, QPalette::Base, brush);
        palette3.setBrush(QPalette::Active, QPalette::Window, brush9);
        palette3.setBrush(QPalette::Inactive, QPalette::WindowText, brush10);
        palette3.setBrush(QPalette::Inactive, QPalette::Button, brush3);
        palette3.setBrush(QPalette::Inactive, QPalette::Light, brush4);
        palette3.setBrush(QPalette::Inactive, QPalette::Midlight, brush5);
        palette3.setBrush(QPalette::Inactive, QPalette::Dark, brush6);
        palette3.setBrush(QPalette::Inactive, QPalette::Text, brush7);
        palette3.setBrush(QPalette::Inactive, QPalette::ButtonText, brush8);
        palette3.setBrush(QPalette::Inactive, QPalette::Base, brush);
        palette3.setBrush(QPalette::Inactive, QPalette::Window, brush9);
        palette3.setBrush(QPalette::Disabled, QPalette::WindowText, brush6);
        palette3.setBrush(QPalette::Disabled, QPalette::Button, brush3);
        palette3.setBrush(QPalette::Disabled, QPalette::Light, brush4);
        palette3.setBrush(QPalette::Disabled, QPalette::Midlight, brush5);
        palette3.setBrush(QPalette::Disabled, QPalette::Dark, brush6);
        palette3.setBrush(QPalette::Disabled, QPalette::Text, brush6);
        palette3.setBrush(QPalette::Disabled, QPalette::ButtonText, brush6);
        palette3.setBrush(QPalette::Disabled, QPalette::Base, brush9);
        palette3.setBrush(QPalette::Disabled, QPalette::Window, brush9);
        averageFrequency->setPalette(palette3);
        averageFrequency->setSmallDecimalPoint(true);
        averageFrequency->setDigitCount(4);
        averageFrequency->setSegmentStyle(QLCDNumber::Filled);
        averageFrequency->setProperty("value", QVariant(60));
        averageFrequency->setProperty("intValue", QVariant(60));
        currentDial = new XqwtMultiDial(pwr);
        currentDial->setObjectName(QStringLiteral("currentDial"));
        currentDial->setGeometry(QRect(10, 10, 261, 181));
        QPalette palette4;
        QBrush brush11(QColor(224, 222, 219, 255));
        brush11.setStyle(Qt::SolidPattern);
        palette4.setBrush(QPalette::Active, QPalette::WindowText, brush11);
        QBrush brush12(QColor(243, 240, 236, 255));
        brush12.setStyle(Qt::SolidPattern);
        palette4.setBrush(QPalette::Active, QPalette::Button, brush12);
        QBrush brush13(QColor(244, 238, 238, 255));
        brush13.setStyle(Qt::SolidPattern);
        palette4.setBrush(QPalette::Active, QPalette::Light, brush13);
        QBrush brush14(QColor(229, 221, 221, 255));
        brush14.setStyle(Qt::SolidPattern);
        palette4.setBrush(QPalette::Active, QPalette::Dark, brush14);
        palette4.setBrush(QPalette::Active, QPalette::Text, brush2);
        palette4.setBrush(QPalette::Active, QPalette::BrightText, brush2);
        palette4.setBrush(QPalette::Active, QPalette::ButtonText, brush8);
        QBrush brush15(QColor(253, 246, 246, 255));
        brush15.setStyle(Qt::SolidPattern);
        palette4.setBrush(QPalette::Active, QPalette::Base, brush15);
        QBrush brush16(QColor(232, 225, 225, 255));
        brush16.setStyle(Qt::SolidPattern);
        palette4.setBrush(QPalette::Active, QPalette::Window, brush16);
        QBrush brush17(QColor(240, 234, 234, 255));
        brush17.setStyle(Qt::SolidPattern);
        palette4.setBrush(QPalette::Active, QPalette::AlternateBase, brush17);
        QBrush brush18(QColor(244, 244, 244, 255));
        brush18.setStyle(Qt::SolidPattern);
        palette4.setBrush(QPalette::Active, QPalette::NoRole, brush18);
        palette4.setBrush(QPalette::Inactive, QPalette::WindowText, brush11);
        palette4.setBrush(QPalette::Inactive, QPalette::Button, brush12);
        palette4.setBrush(QPalette::Inactive, QPalette::Light, brush13);
        palette4.setBrush(QPalette::Inactive, QPalette::Dark, brush14);
        palette4.setBrush(QPalette::Inactive, QPalette::Text, brush2);
        palette4.setBrush(QPalette::Inactive, QPalette::BrightText, brush2);
        palette4.setBrush(QPalette::Inactive, QPalette::ButtonText, brush8);
        palette4.setBrush(QPalette::Inactive, QPalette::Base, brush15);
        palette4.setBrush(QPalette::Inactive, QPalette::Window, brush16);
        palette4.setBrush(QPalette::Inactive, QPalette::AlternateBase, brush17);
        palette4.setBrush(QPalette::Inactive, QPalette::NoRole, brush18);
        palette4.setBrush(QPalette::Disabled, QPalette::WindowText, brush14);
        palette4.setBrush(QPalette::Disabled, QPalette::Button, brush12);
        palette4.setBrush(QPalette::Disabled, QPalette::Light, brush13);
        palette4.setBrush(QPalette::Disabled, QPalette::Dark, brush14);
        palette4.setBrush(QPalette::Disabled, QPalette::Text, brush14);
        palette4.setBrush(QPalette::Disabled, QPalette::BrightText, brush2);
        palette4.setBrush(QPalette::Disabled, QPalette::ButtonText, brush14);
        palette4.setBrush(QPalette::Disabled, QPalette::Base, brush16);
        palette4.setBrush(QPalette::Disabled, QPalette::Window, brush16);
        palette4.setBrush(QPalette::Disabled, QPalette::AlternateBase, brush17);
        palette4.setBrush(QPalette::Disabled, QPalette::NoRole, brush18);
        currentDial->setPalette(palette4);
        gridLayoutWidget = new QWidget(pwr);
        gridLayoutWidget->setObjectName(QStringLiteral("gridLayoutWidget"));
        gridLayoutWidget->setGeometry(QRect(10, 260, 701, 141));
        switchGrid = new QGridLayout(gridLayoutWidget);
        switchGrid->setObjectName(QStringLiteral("switchGrid"));
        switchGrid->setContentsMargins(0, 0, 0, 0);
        voltageDial = new XqwtMultiDial(pwr);
        voltageDial->setObjectName(QStringLiteral("voltageDial"));
        voltageDial->setGeometry(QRect(250, 10, 261, 181));
        QPalette palette5;
        palette5.setBrush(QPalette::Active, QPalette::WindowText, brush11);
        palette5.setBrush(QPalette::Active, QPalette::Button, brush12);
        palette5.setBrush(QPalette::Active, QPalette::Light, brush13);
        palette5.setBrush(QPalette::Active, QPalette::Dark, brush14);
        palette5.setBrush(QPalette::Active, QPalette::Text, brush2);
        palette5.setBrush(QPalette::Active, QPalette::BrightText, brush2);
        palette5.setBrush(QPalette::Active, QPalette::ButtonText, brush8);
        palette5.setBrush(QPalette::Active, QPalette::Base, brush15);
        palette5.setBrush(QPalette::Active, QPalette::Window, brush16);
        palette5.setBrush(QPalette::Active, QPalette::AlternateBase, brush17);
        palette5.setBrush(QPalette::Active, QPalette::NoRole, brush18);
        palette5.setBrush(QPalette::Inactive, QPalette::WindowText, brush11);
        palette5.setBrush(QPalette::Inactive, QPalette::Button, brush12);
        palette5.setBrush(QPalette::Inactive, QPalette::Light, brush13);
        palette5.setBrush(QPalette::Inactive, QPalette::Dark, brush14);
        palette5.setBrush(QPalette::Inactive, QPalette::Text, brush2);
        palette5.setBrush(QPalette::Inactive, QPalette::BrightText, brush2);
        palette5.setBrush(QPalette::Inactive, QPalette::ButtonText, brush8);
        palette5.setBrush(QPalette::Inactive, QPalette::Base, brush15);
        palette5.setBrush(QPalette::Inactive, QPalette::Window, brush16);
        palette5.setBrush(QPalette::Inactive, QPalette::AlternateBase, brush17);
        palette5.setBrush(QPalette::Inactive, QPalette::NoRole, brush18);
        palette5.setBrush(QPalette::Disabled, QPalette::WindowText, brush14);
        palette5.setBrush(QPalette::Disabled, QPalette::Button, brush12);
        palette5.setBrush(QPalette::Disabled, QPalette::Light, brush13);
        palette5.setBrush(QPalette::Disabled, QPalette::Dark, brush14);
        palette5.setBrush(QPalette::Disabled, QPalette::Text, brush14);
        palette5.setBrush(QPalette::Disabled, QPalette::BrightText, brush2);
        palette5.setBrush(QPalette::Disabled, QPalette::ButtonText, brush14);
        palette5.setBrush(QPalette::Disabled, QPalette::Base, brush16);
        palette5.setBrush(QPalette::Disabled, QPalette::Window, brush16);
        palette5.setBrush(QPalette::Disabled, QPalette::AlternateBase, brush17);
        palette5.setBrush(QPalette::Disabled, QPalette::NoRole, brush18);
        voltageDial->setPalette(palette5);
        frequencyDial = new XqwtMultiDial(pwr);
        frequencyDial->setObjectName(QStringLiteral("frequencyDial"));
        frequencyDial->setGeometry(QRect(490, 10, 261, 181));
        QPalette palette6;
        palette6.setBrush(QPalette::Active, QPalette::WindowText, brush11);
        palette6.setBrush(QPalette::Active, QPalette::Button, brush12);
        palette6.setBrush(QPalette::Active, QPalette::Light, brush13);
        palette6.setBrush(QPalette::Active, QPalette::Dark, brush14);
        palette6.setBrush(QPalette::Active, QPalette::Text, brush2);
        palette6.setBrush(QPalette::Active, QPalette::BrightText, brush2);
        palette6.setBrush(QPalette::Active, QPalette::ButtonText, brush8);
        palette6.setBrush(QPalette::Active, QPalette::Base, brush15);
        palette6.setBrush(QPalette::Active, QPalette::Window, brush16);
        palette6.setBrush(QPalette::Active, QPalette::AlternateBase, brush17);
        palette6.setBrush(QPalette::Active, QPalette::NoRole, brush18);
        palette6.setBrush(QPalette::Inactive, QPalette::WindowText, brush11);
        palette6.setBrush(QPalette::Inactive, QPalette::Button, brush12);
        palette6.setBrush(QPalette::Inactive, QPalette::Light, brush13);
        palette6.setBrush(QPalette::Inactive, QPalette::Dark, brush14);
        palette6.setBrush(QPalette::Inactive, QPalette::Text, brush2);
        palette6.setBrush(QPalette::Inactive, QPalette::BrightText, brush2);
        palette6.setBrush(QPalette::Inactive, QPalette::ButtonText, brush8);
        palette6.setBrush(QPalette::Inactive, QPalette::Base, brush15);
        palette6.setBrush(QPalette::Inactive, QPalette::Window, brush16);
        palette6.setBrush(QPalette::Inactive, QPalette::AlternateBase, brush17);
        palette6.setBrush(QPalette::Inactive, QPalette::NoRole, brush18);
        palette6.setBrush(QPalette::Disabled, QPalette::WindowText, brush14);
        palette6.setBrush(QPalette::Disabled, QPalette::Button, brush12);
        palette6.setBrush(QPalette::Disabled, QPalette::Light, brush13);
        palette6.setBrush(QPalette::Disabled, QPalette::Dark, brush14);
        palette6.setBrush(QPalette::Disabled, QPalette::Text, brush14);
        palette6.setBrush(QPalette::Disabled, QPalette::BrightText, brush2);
        palette6.setBrush(QPalette::Disabled, QPalette::ButtonText, brush14);
        palette6.setBrush(QPalette::Disabled, QPalette::Base, brush16);
        palette6.setBrush(QPalette::Disabled, QPalette::Window, brush16);
        palette6.setBrush(QPalette::Disabled, QPalette::AlternateBase, brush17);
        palette6.setBrush(QPalette::Disabled, QPalette::NoRole, brush18);
        frequencyDial->setPalette(palette6);
        ampsLabel = new QLabel(pwr);
        ampsLabel->setObjectName(QStringLiteral("ampsLabel"));
        ampsLabel->setGeometry(QRect(30, 177, 91, 24));
        QPalette palette7;
        palette7.setBrush(QPalette::Active, QPalette::WindowText, brush2);
        palette7.setBrush(QPalette::Inactive, QPalette::WindowText, brush2);
        QBrush brush19(QColor(96, 95, 94, 255));
        brush19.setStyle(Qt::SolidPattern);
        palette7.setBrush(QPalette::Disabled, QPalette::WindowText, brush19);
        ampsLabel->setPalette(palette7);
        ampsLabel->setAlignment(Qt::AlignCenter);
        ampsLabel_2 = new QLabel(pwr);
        ampsLabel_2->setObjectName(QStringLiteral("ampsLabel_2"));
        ampsLabel_2->setGeometry(QRect(240, 177, 101, 24));
        QPalette palette8;
        palette8.setBrush(QPalette::Active, QPalette::WindowText, brush2);
        palette8.setBrush(QPalette::Inactive, QPalette::WindowText, brush2);
        palette8.setBrush(QPalette::Disabled, QPalette::WindowText, brush19);
        ampsLabel_2->setPalette(palette8);
        ampsLabel_2->setAlignment(Qt::AlignCenter);
        ampsLabel_3 = new QLabel(pwr);
        ampsLabel_3->setObjectName(QStringLiteral("ampsLabel_3"));
        ampsLabel_3->setGeometry(QRect(490, 177, 101, 24));
        QPalette palette9;
        palette9.setBrush(QPalette::Active, QPalette::WindowText, brush2);
        palette9.setBrush(QPalette::Inactive, QPalette::WindowText, brush2);
        palette9.setBrush(QPalette::Disabled, QPalette::WindowText, brush19);
        ampsLabel_3->setPalette(palette9);
        ampsLabel_3->setAlignment(Qt::AlignCenter);
        currentDial->raise();
        totalCurrent->raise();
        averageVoltage->raise();
        averageFrequency->raise();
        gridLayoutWidget->raise();
        voltageDial->raise();
        frequencyDial->raise();
        ampsLabel->raise();
        ampsLabel_2->raise();
        ampsLabel_3->raise();

        retranslateUi(pwr);

        QMetaObject::connectSlotsByName(pwr);
    } // setupUi

    void retranslateUi(QWidget *pwr)
    {
        pwr->setWindowTitle(QApplication::translate("pwr", "Form", Q_NULLPTR));
        ampsLabel->setText(QApplication::translate("pwr", "Tot. Amps:", Q_NULLPTR));
        ampsLabel_2->setText(QApplication::translate("pwr", "Avg. Volts:", Q_NULLPTR));
        ampsLabel_3->setText(QApplication::translate("pwr", "Avg. Hz:", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class pwr: public Ui_pwr {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PWR_H
