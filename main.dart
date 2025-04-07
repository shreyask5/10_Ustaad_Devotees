import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'screens/home_screen.dart';
import 'screens/sos_screen.dart';
import 'screens/guidance_screen.dart';
import 'screens/safety_map_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(const WomenSafetyApp());
}

class WomenSafetyApp extends StatelessWidget {
  const WomenSafetyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Women Safety App',
      theme: ThemeData(
        primarySwatch: Colors.pink,
        useMaterial3: true,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      initialRoute: '/',
      routes: {
        '/': (context) => const HomeScreen(),
        '/sos': (context) => const SOSScreen(),
        '/guidance': (context) => const GuidanceScreen(),
        '/safety-map': (context) => const SafetyMapScreen(),
      },
    );
  }
} 