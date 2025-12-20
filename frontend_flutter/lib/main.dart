import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

String server = "192.168.168.19:8000"; // 伺服器ip和port
const keyNames = {
  "0": "C",
  "1": "C#/Db",
  "2": "D",
  "3": "D#/Eb",
  "4": "E",
  "5": "F",
  "6": "F#/Gb",
  "7": "G",
  "8": "G#/Ab",
  "9": "A",
  "10": "A#/Bb",
  "11": "B",
};
void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '推歌之王',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: Center(child: const Text('推歌之王')),
          bottom: const TabBar(
            tabs: [
              Tab(text: '上傳音檔'),
              Tab(text: '歷史紀錄'),
            ],
          ),
        ),
        body: const TabBarView(children: [UploadPage(), HistoryPage()]),
      ),
    );
  }
}

// ================= UploadPage =================

class UploadPage extends StatefulWidget {
  const UploadPage({super.key});

  @override
  State<UploadPage> createState() => _UploadPageState();
}

class _UploadPageState extends State<UploadPage> {
  PlatformFile? selectedFile;
  bool isLoading = false;

  Future<List<Map<String, dynamic>>> analyzeFile(File file) async {
    setState(() => isLoading = true);

    try {
      var uri = Uri.http(server, "/upload");
      var request = http.MultipartRequest("POST", uri);

      request.files.add(await http.MultipartFile.fromPath("file", file.path));

      var streamedResponse = await request.send(); //送音檔給後端並等待後端回傳資料
      var response = await http.Response.fromStream(
        streamedResponse,
      ); //把stream轉成完整的response
      //大小為5的list裡面放json
      /*
      "artist_name": hdf5_getters.get_artist_name(h5).decode(),
      "title": hdf5_getters.get_title(h5).decode(),
      "year": int(hdf5_getters.get_year(h5)),

      "duration": float(hdf5_getters.get_duration(h5)),
      "tempo": float(hdf5_getters.get_tempo(h5)),
      "loudness": float(hdf5_getters.get_loudness(h5)),
      "key": int(hdf5_getters.get_key(h5)),
      "mode": int(hdf5_getters.get_mode(h5)),

      "timbre_mean": timbre_mean,           //shape(12,)
      "timbre_std": timbre_std,             //shape(12,)
      "pitches_mean": pitches_mean,         //shape(12,)
      "pitches_std": pitches_std            //shape(12,)
      */

      if (response.statusCode == 200) {
        List<dynamic> jsonList = jsonDecode(response.body);
        List<Map<String, dynamic>> results = jsonList
            .map((e) => e as Map<String, dynamic>)
            .toList();

        // 存歷史紀錄：上傳檔名 + 五首歌資料
        await HistoryStorage.addRecord(file.path.split("/").last, results);

        return results;
      } else {
        throw Exception("上傳失敗，HTTP 狀態碼: ${response.statusCode}");
      }
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("連線失敗，請檢查網路或後端是否啟動: $e")));
      throw Exception("連線失敗，請檢查網路或後端是否啟動");
    } finally {
      if (mounted) setState(() => isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("上傳音檔")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (selectedFile != null) Text("已選擇：${selectedFile!.name}"),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () async {
                FilePickerResult? result = await FilePicker.platform.pickFiles(
                  type: FileType.custom,
                  allowedExtensions: ["mp3", "wav", "m4a"],
                );
                if (result != null) {
                  setState(() {
                    selectedFile = result.files.single;
                  });
                }
              },
              child: const Text("選擇音檔"),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: selectedFile == null || isLoading
                  ? null
                  : () async {
                      File file = File(selectedFile!.path!);
                      List<Map<String, dynamic>> results = await analyzeFile(
                        file,
                      );
                      if (!mounted) return;
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => ResultPage(results: results),
                        ),
                      );
                    },
              child: isLoading
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        color: Colors.white,
                        strokeWidth: 2,
                      ),
                    )
                  : const Text("分析"),
            ),
          ],
        ),
      ),
    );
  }
}

// ================= ResultPage =================

class ResultPage extends StatelessWidget {
  final List<Map<String, dynamic>> results;

  const ResultPage({super.key, required this.results});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("分析結果")),
      body: ListView.builder(
        itemCount: results.length, //itemCount是多少下面itemBuilder就自動呼叫幾次，5首歌5次
        itemBuilder: (context, index) {
          var song = results[index];
          return ListTile(
            title: Text(song["title"]), //歌名
            subtitle: Text(song["result"] ?? ""), //相似度
            onTap: () {
              //點擊看詳細資料
              Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => DetailPage(song: song)),
              );
            },
          );
        },
      ),
    );
  }
}

// ================= HistoryStorage =================

class HistoryStorage {
  static const key = "history";

  static Future<void> addRecord(
    String uploadedFile, // 上傳的檔名
    List<Map<String, dynamic>> results, // 後端回傳的五首相似歌曲資料
  ) async {
    // 取得 SharedPreferences 實例，用來存取本地資料
    final prefs = await SharedPreferences.getInstance();

    // 讀取現有的歷史紀錄列表，如果不存在就建立空列表
    List<String> list = prefs.getStringList(key) ?? [];

    // 建立一筆新的紀錄，包含上傳檔名、日期以及五首歌的資料
    Map<String, dynamic> record = {
      "uploaded_file": uploadedFile, // 上傳檔名
      "date": DateTime.now().toString().substring(
        0,
        10,
      ), // 自動取得當天日期 (yyyy-MM-dd)
      "results": results, // 五首最相近歌曲的資料列表
    };

    // 把這筆紀錄轉成 JSON 字串並加入列表
    list.add(jsonEncode(record));

    // 將更新後的列表存回 SharedPreferences
    await prefs.setStringList(key, list);
  }

  static Future<List<Map<String, dynamic>>> getRecords() async {
    final prefs = await SharedPreferences.getInstance();
    List<String> list = prefs.getStringList(key) ?? [];
    return list.map((e) => jsonDecode(e) as Map<String, dynamic>).toList();
  }
}

// ================= HistoryPage =================

class HistoryPage extends StatefulWidget {
  const HistoryPage({super.key});

  @override
  State<HistoryPage> createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  List<Map<String, dynamic>> history = [];

  @override
  void initState() {
    super.initState();
    loadHistory();
  }

  Future<void> loadHistory() async {
    history = await HistoryStorage.getRecords();
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("歷史紀錄")),
      body: ListView.builder(
        itemCount: history.length,
        itemBuilder: (context, index) {
          final item = history[index];
          return ListTile(
            title: Text("${item["uploaded_file"]}"),
            subtitle: Text(item["date"]),
            onTap: () {
              // 點擊跳到 ResultPage 顯示五首歌資料
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => ResultPage(
                    results: List<Map<String, dynamic>>.from(
                      item["results"] ?? [],
                    ),
                  ),
                ),
              );
            },
          );
        },
      ),
    );
  }
}

// ================= DetailPage =================

class DetailPage extends StatelessWidget {
  final Map<String, dynamic> song;

  const DetailPage({super.key, required this.song});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(song["file"])),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text("歌名：${song["title"] ?? ""}"),
              Text("作者：${song["artist_name"] ?? ""}"),
              Text(
                "年份：${song["year"].toString() != "0" ? song["year"] : "未知"}",
              ),
              Text("長度：${song["duration"] ?? ""} 秒"),
              Text("節奏（Tempo）：${song["tempo"] ?? ""}"),
              Text("音量（Loudness）：${song["loudness"] ?? ""}"),
              Text("調性（Key）：${keyNames[song["key"].toString()]}"),
              Text("調式：${song["mode"].toString() == "1" ? "大調" : "小調"}"),
              //Text("音色(Timbre) 平均值：${song["timbre_mean"] ?? ""}"),
              //Text("音色(Timbre) 標準差：${song["timbre_std"] ?? ""}"),
              //Text("音高(Pitches) 平均值：${song["pitches_mean"] ?? ""}"),
              //Text("音高(Pitches) 標準差：${song["pitches_std"] ?? ""}"),
              Text("相似度結果：${song["result"] ?? ""}"),
            ],
          ),
        ),
      ),
    );
  }
}
