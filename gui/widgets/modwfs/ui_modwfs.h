/********************************************************************************
** Form generated from reading UI file 'modwfs.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MODWFS_H
#define UI_MODWFS_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLCDNumber>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_modwfs
{
public:
    QLabel *titleLabel;
    QWidget *gridLayoutWidget;
    QGridLayout *gridLayout_2;
    QPushButton *buttonRest;
    QPushButton *buttonSet;
    QPushButton *buttonModulate;
    QPushButton *buttonUp;
    QPushButton *buttonLeft;
    QPushButton *buttonRight;
    QPushButton *buttonDown;
    QLCDNumber *voltsAxis1;
    QLCDNumber *voltsAxis2;
    QLabel *labelChannel1;
    QLabel *labelChannel2;
    QLabel *ttmStatus;

    void setupUi(QWidget *modwfs)
    {
        if (modwfs->objectName().isEmpty())
            modwfs->setObjectName(QStringLiteral("modwfs"));
        modwfs->resize(512, 509);
        QPalette palette;
        QBrush brush(QColor(32, 31, 31, 255));
        brush.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Base, brush);
        QBrush brush1(QColor(0, 0, 0, 255));
        brush1.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Window, brush1);
        palette.setBrush(QPalette::Inactive, QPalette::Base, brush);
        palette.setBrush(QPalette::Inactive, QPalette::Window, brush1);
        palette.setBrush(QPalette::Disabled, QPalette::Base, brush1);
        palette.setBrush(QPalette::Disabled, QPalette::Window, brush1);
        modwfs->setPalette(palette);
        modwfs->setFocusPolicy(Qt::StrongFocus);
        titleLabel = new QLabel(modwfs);
        titleLabel->setObjectName(QStringLiteral("titleLabel"));
        titleLabel->setGeometry(QRect(10, 20, 491, 24));
        titleLabel->setAlignment(Qt::AlignCenter);
        gridLayoutWidget = new QWidget(modwfs);
        gridLayoutWidget->setObjectName(QStringLiteral("gridLayoutWidget"));
        gridLayoutWidget->setGeometry(QRect(10, 150, 491, 88));
        gridLayout_2 = new QGridLayout(gridLayoutWidget);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        buttonRest = new QPushButton(gridLayoutWidget);
        buttonRest->setObjectName(QStringLiteral("buttonRest"));
        buttonRest->setFocusPolicy(Qt::StrongFocus);

        gridLayout_2->addWidget(buttonRest, 0, 0, 1, 1);

        buttonSet = new QPushButton(gridLayoutWidget);
        buttonSet->setObjectName(QStringLiteral("buttonSet"));
        buttonSet->setFocusPolicy(Qt::StrongFocus);

        gridLayout_2->addWidget(buttonSet, 0, 1, 1, 1);

        buttonModulate = new QPushButton(gridLayoutWidget);
        buttonModulate->setObjectName(QStringLiteral("buttonModulate"));
        buttonModulate->setFocusPolicy(Qt::StrongFocus);

        gridLayout_2->addWidget(buttonModulate, 0, 2, 1, 1);

        buttonUp = new QPushButton(modwfs);
        buttonUp->setObjectName(QStringLiteral("buttonUp"));
        buttonUp->setGeometry(QRect(140, 260, 41, 40));
        buttonUp->setFocusPolicy(Qt::StrongFocus);
        buttonLeft = new QPushButton(modwfs);
        buttonLeft->setObjectName(QStringLiteral("buttonLeft"));
        buttonLeft->setGeometry(QRect(100, 300, 41, 40));
        buttonLeft->setFocusPolicy(Qt::StrongFocus);
        buttonRight = new QPushButton(modwfs);
        buttonRight->setObjectName(QStringLiteral("buttonRight"));
        buttonRight->setGeometry(QRect(180, 300, 41, 40));
        buttonRight->setFocusPolicy(Qt::StrongFocus);
        buttonDown = new QPushButton(modwfs);
        buttonDown->setObjectName(QStringLiteral("buttonDown"));
        buttonDown->setGeometry(QRect(140, 340, 41, 40));
        buttonDown->setFocusPolicy(Qt::StrongFocus);
        voltsAxis1 = new QLCDNumber(modwfs);
        voltsAxis1->setObjectName(QStringLiteral("voltsAxis1"));
        voltsAxis1->setGeometry(QRect(150, 400, 101, 41));
        QPalette palette1;
        QBrush brush2(QColor(170, 255, 255, 255));
        brush2.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::WindowText, brush2);
        palette1.setBrush(QPalette::Active, QPalette::Text, brush2);
        palette1.setBrush(QPalette::Inactive, QPalette::WindowText, brush2);
        palette1.setBrush(QPalette::Inactive, QPalette::Text, brush2);
        QBrush brush3(QColor(96, 95, 94, 255));
        brush3.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Disabled, QPalette::WindowText, brush3);
        QBrush brush4(QColor(83, 82, 81, 255));
        brush4.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Disabled, QPalette::Text, brush4);
        voltsAxis1->setPalette(palette1);
        voltsAxis2 = new QLCDNumber(modwfs);
        voltsAxis2->setObjectName(QStringLiteral("voltsAxis2"));
        voltsAxis2->setGeometry(QRect(150, 450, 101, 41));
        QPalette palette2;
        palette2.setBrush(QPalette::Active, QPalette::WindowText, brush2);
        palette2.setBrush(QPalette::Inactive, QPalette::WindowText, brush2);
        palette2.setBrush(QPalette::Disabled, QPalette::WindowText, brush3);
        voltsAxis2->setPalette(palette2);
        labelChannel1 = new QLabel(modwfs);
        labelChannel1->setObjectName(QStringLiteral("labelChannel1"));
        labelChannel1->setGeometry(QRect(5, 410, 131, 24));
        QPalette palette3;
        QBrush brush5(QColor(0, 0, 102, 255));
        brush5.setStyle(Qt::SolidPattern);
        palette3.setBrush(QPalette::Active, QPalette::Text, brush5);
        palette3.setBrush(QPalette::Inactive, QPalette::Text, brush5);
        palette3.setBrush(QPalette::Disabled, QPalette::Text, brush4);
        labelChannel1->setPalette(palette3);
        QFont font;
        font.setFamily(QStringLiteral("Tlwg Typewriter"));
        font.setPointSize(14);
        labelChannel1->setFont(font);
        labelChannel2 = new QLabel(modwfs);
        labelChannel2->setObjectName(QStringLiteral("labelChannel2"));
        labelChannel2->setGeometry(QRect(5, 460, 131, 24));
        labelChannel2->setFont(font);
        ttmStatus = new QLabel(modwfs);
        ttmStatus->setObjectName(QStringLiteral("ttmStatus"));
        ttmStatus->setGeometry(QRect(20, 80, 471, 51));
        QPalette palette4;
        palette4.setBrush(QPalette::Active, QPalette::WindowText, brush2);
        palette4.setBrush(QPalette::Inactive, QPalette::WindowText, brush2);
        palette4.setBrush(QPalette::Disabled, QPalette::WindowText, brush3);
        ttmStatus->setPalette(palette4);
        QFont font1;
        font1.setFamily(QStringLiteral("Tlwg Typist"));
        font1.setPointSize(22);
        ttmStatus->setFont(font1);
        ttmStatus->setFrameShape(QFrame::Box);
        ttmStatus->setFrameShadow(QFrame::Sunken);
        ttmStatus->setAlignment(Qt::AlignCenter);

        retranslateUi(modwfs);

        QMetaObject::connectSlotsByName(modwfs);
    } // setupUi

    void retranslateUi(QWidget *modwfs)
    {
        modwfs->setWindowTitle(QApplication::translate("modwfs", "modwfs", Q_NULLPTR));
        titleLabel->setText(QApplication::translate("modwfs", "PyWFS Modulator", Q_NULLPTR));
        buttonRest->setText(QApplication::translate("modwfs", "Rest", Q_NULLPTR));
        buttonSet->setText(QApplication::translate("modwfs", "Set", Q_NULLPTR));
        buttonModulate->setText(QApplication::translate("modwfs", "Modulate", Q_NULLPTR));
        buttonUp->setText(QApplication::translate("modwfs", "U", Q_NULLPTR));
        buttonLeft->setText(QApplication::translate("modwfs", "L", Q_NULLPTR));
        buttonRight->setText(QApplication::translate("modwfs", "R", Q_NULLPTR));
        buttonDown->setText(QApplication::translate("modwfs", "D", Q_NULLPTR));
        labelChannel1->setText(QApplication::translate("modwfs", "Channel 1:", Q_NULLPTR));
        labelChannel2->setText(QApplication::translate("modwfs", "Channel 2:", Q_NULLPTR));
        ttmStatus->setText(QApplication::translate("modwfs", "OFF", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class modwfs: public Ui_modwfs {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MODWFS_H
